"""Deterministic CE/KL smoke training, FSDP2 integration and exact resume check.

Synthetic/local fixtures establish engineering correctness, never model quality.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import random
import time

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict, StateDictOptions
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from transformers import Qwen3Config, Qwen3ForCausalLM, AutoTokenizer
import yaml
from .checkpoint import from_qwen_model, load_qwen_as_rmt
from .data import pack_sequences, FIXTURES
from .runtime import enforce_gpu_scope, write_report, versions


def local(parameter):
    return parameter.to_local() if hasattr(parameter, 'to_local') else parameter


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    args=parser.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text())
    enforce_gpu_scope()
    rank=int(os.environ.get('RANK',0)); world=int(os.environ.get('WORLD_SIZE',1))
    device=torch.device('cuda',int(os.environ.get('LOCAL_RANK',0)))
    torch.cuda.set_device(device); torch.set_num_threads(2)
    if world>1: dist.init_process_group('nccl')
    torch.manual_seed(cfg.get('seed',17)); random.seed(cfg.get('seed',17))
    full=cfg.get('full_model',False)
    dtype=torch.bfloat16 if cfg.get('bf16',False) else torch.float32
    if full:
        model,_=load_qwen_as_rmt(cfg['base_model'],dtype=torch.float32)
        teacher,_=load_qwen_as_rmt(cfg['base_model'],dtype=dtype)
        tokenizer=AutoTokenizer.from_pretrained(cfg['base_model'],local_files_only=True)
        docs=[tokenizer.encode(x,add_special_tokens=False)[:cfg['sequence_length']//2] for x in FIXTURES]
        pad=tokenizer.pad_token_id
    else:
        config=Qwen3Config(vocab_size=97,hidden_size=32,intermediate_size=48,num_hidden_layers=3,
            num_attention_heads=4,num_key_value_heads=2,head_dim=8,max_position_embeddings=128,
            tie_word_embeddings=True,pad_token_id=0,eos_token_id=2)
        config._attn_implementation='eager'
        original=Qwen3ForCausalLM(config)
        model=from_qwen_model(original)
        teacher=copy.deepcopy(model).to(dtype=dtype)
        docs=[[5,7,10,20],[12,21,35,9],[24,17,22,6],[8,3,15,40]]; pad=0
    model.config.routing_mode='learned'
    model.model.cell.router.prior_strength=cfg.get('prior_strength',0.0)
    with torch.no_grad(): model.model.cell.router.weight.normal_(std=0.01)
    teacher=teacher.to(device).eval().requires_grad_(False)
    model=model.to(device).train()
    if cfg.get('checkpoint',True): model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    if cfg.get('compile',False): model.model.cell.bank.compile_projections()
    # One stable root communication boundary: conditional expert calls never initiate collectives.
    if world>1:
        fully_shard(model,reshard_after_forward=False,
            mp_policy=MixedPrecisionPolicy(param_dtype=dtype,reduce_dtype=torch.float32))
    elif dtype!=torch.float32:
        # BF16 autocast retains FP32 master weights in non-sharded mode.
        pass
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg.get('learning_rate',1e-3),weight_decay=0.0,foreach=False)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:1.0/(1.0+step/100))
    outdir=Path(cfg['artifact_dir']); outdir.mkdir(parents=True,exist_ok=True)
    rows=[]; cursor=0
    def batch_for(step,which_rank):
        selected=[docs[(step+which_rank)%len(docs)], docs[(step+which_rank+1)%len(docs)]]
        return pack_sequences(selected,pad,cfg['sequence_length'],device)
    def step_once(step):
        batch=batch_for(step,rank)
        optimizer.zero_grad(set_to_none=True)
        prior=cfg.get('prior_strength',0.0)*max(0,1-step/max(cfg['steps'],1))
        model.model.cell.router.prior_strength=prior
        start=time.monotonic()
        with torch.no_grad():
            teacher_hidden=teacher(**{k:v for k,v in batch.items() if k!='labels'},
                use_cache=False,return_hidden_only=True).last_hidden_state
        # Final controlled step exercises repeated mixed routes and globally unused experts.
        forced=cfg.get('forced_last_step',True) and step==cfg['steps']-1
        routing={}
        if forced:
            tokens=torch.arange(batch['input_ids'].numel(),device=device).view_as(batch['input_ids'])
            routes=(tokens.unsqueeze(-1)+rank+torch.arange(model.config.num_recurrences,device=device))%min(2,model.config.num_experts)
            routing={'routing_mode':'forced','forced_routes':routes.long()}
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=dtype==torch.bfloat16):
            output=model(**batch,use_cache=False,loss_chunk_size=cfg.get('loss_chunk_size',8),
                teacher_hidden_states=teacher_hidden,teacher_head_weight=teacher.lm_head.weight,
                kd_weight=cfg.get('kd_weight',0.5),output_router_trace=True,**routing)
        assert torch.isfinite(output.loss), 'nonfinite loss'
        output.loss.backward()
        grad=local(model.model.cell.router.weight.grad) if model.model.cell.router.weight.grad is not None else None
        router_norm=0.0 if grad is None else grad.float().norm().item()
        for p in model.parameters():
            if p.grad is not None: assert torch.isfinite(local(p.grad)).all(), 'nonfinite gradient'
        # No rank-local clipping: it would change the sharded model's update.
        optimizer.step(); scheduler.step(); torch.cuda.synchronize()
        valid=batch['attention_mask'].bool()
        counts=torch.zeros(model.config.num_experts,device=device,dtype=torch.long)
        for chosen in output.router_indices: counts+=torch.bincount(chosen[valid],minlength=counts.numel())
        values=torch.tensor([output.loss.item(),output.ce_loss.item(),output.kd_loss.item(),router_norm],device=device)
        if world>1: dist.all_reduce(values); values/=world; dist.all_reduce(counts)
        elapsed=time.monotonic()-start
        tokens=int(valid.sum())*world
        row={'step':step,'loss':values[0].item(),'ce':values[1].item(),'kl':values[2].item(),
            'router_grad_norm_mean_local':values[3].item(),'routes':counts.tolist(),
            'mode':'forced' if forced else 'learned','prior_strength':prior,
            'seconds':elapsed,'valid_tokens_per_second':tokens/elapsed,
            'peak_memory_bytes':torch.cuda.max_memory_allocated(device)}
        if rank==0: print(json.dumps(row),flush=True)
        return row
    for cursor in range(cfg['steps']): rows.append(step_once(cursor))
    # Save at an optimizer-step boundary; each rank records its own RNG and cursor.
    options=StateDictOptions(cpu_offload=True)
    state_model,state_opt=get_state_dict(model,optimizer,options=options)
    state={'model':state_model,'optimizer':state_opt}
    dcp.save(state,checkpoint_id=outdir/'resume')
    rng={'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state(device),'python':random.getstate(),
         'scheduler':scheduler.state_dict(),'cursor':cfg['steps'],
         'prior_strength':model.model.cell.router.prior_strength,'config':cfg}
    torch.save(rng,outdir/f'rank_{rank}_runtime.pt')
    del state,state_model,state_opt
    reference_row=step_once(cfg['steps'])
    reference={name:local(p).detach().cpu().clone() for name,p in model.named_parameters()}
    state_model,state_opt=get_state_dict(model,optimizer,options=options)
    state={'model':state_model,'optimizer':state_opt}
    dcp.load(state,checkpoint_id=outdir/'resume')
    set_state_dict(model,optimizer,model_state_dict=state['model'],optim_state_dict=state['optimizer'],options=options)
    runtime=torch.load(outdir/f'rank_{rank}_runtime.pt',weights_only=False)
    scheduler.load_state_dict(runtime['scheduler']); torch.set_rng_state(runtime['torch'])
    torch.cuda.set_rng_state(runtime['cuda'],device); random.setstate(runtime['python'])
    model.model.cell.router.prior_strength=runtime['prior_strength']
    del state,state_model,state_opt
    restored_row=step_once(runtime['cursor'])
    assert reference_row['loss']==restored_row['loss'], (reference_row,restored_row)
    max_difference=0.0
    for name,p in model.named_parameters():
        current=local(p).detach().cpu()
        error=(reference[name]-current).abs().max().item() if current.numel() else 0.0
        max_difference=max(max_difference,error)
    assert max_difference==0.0, f'resume parameter difference {max_difference}'
    from torch._dynamo.utils import counters
    if rank==0:
        write_report(cfg['report'],{'status':'passed','config':cfg,'environment':versions(),'world_size':world,
            'steps':rows,'resume':{'next_loss_equal':True,'max_parameter_difference':max_difference},
            'compile_counters':{str(k):dict(v) for k,v in counters.items()},
            'compile_scope':'QKV/norm and O/FFN kernels; Python token dispatch remains eager',
            'training_scope':'fixtures only; no quality claim'})
    if world>1: dist.destroy_process_group()


if __name__=='__main__': main()

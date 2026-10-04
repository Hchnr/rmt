"""Reproducible packed CE/KL training with FSDP2, holdout checks and HF export.

This entry point is separate from the bootstrap stress-test schedule. It never
forces a destructive last step or silently changes the routing mode on resume.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import time
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_state_dict,set_state_dict,StateDictOptions
from torch.distributed.fsdp import fully_shard,MixedPrecisionPolicy
from transformers import AutoTokenizer,Qwen3Config,Qwen3ForCausalLM
import yaml
from .checkpoint import load_qwen_as_rmt,from_qwen_model,export_hf
from .training_data import load_packs,batch_at,file_sha
from .data import pack_sequences
from .losses import shifted_targets
from .halting import prior_expert
from .runtime import enforce_gpu_scope,versions,write_report


def local(tensor):
    return tensor.to_local() if hasattr(tensor,'to_local') else tensor


def prior_at(cfg,step):
    start=cfg.get('prior_start',4.0);end=cfg.get('prior_end',start)
    warm=cfg.get('prior_warmup_steps',0);decay=max(1,cfg.get('prior_decay_steps',cfg['steps']))
    return start+(end-start)*min(1,max(0,(step-warm)/decay))


def parameter_digest(model):
    h=hashlib.sha256()
    for name,p in model.named_parameters():
        h.update(name.encode());h.update(local(p).detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def recurrence_controls(cfg, step):
    """Rank-independent curriculum, reproducible from the saved step and seed."""
    curriculum = cfg.get('recurrence_training', {})
    if step is None or not curriculum:
        return {}
    rng = random.Random(cfg.get('seed',17) * 1000003 + step)
    if step < curriculum.get('fixed_warmup_steps',0):
        return {'halting_policy':'fixed','recurrence_limit':curriculum.get('base_depth',36)}
    if curriculum.get('mode') == 'random':
        depth = rng.choices(curriculum['depths'], curriculum['probabilities'])[0]
        return {'halting_policy':'fixed','recurrence_limit':depth}
    if rng.random() < curriculum.get('forced_fraction',0):
        return {'halting_policy':'fixed','recurrence_limit':cfg['model_overrides']['max_recurrences']}
    return {}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True)
    ap.add_argument('--resume',action='store_true');ap.add_argument('--verify-resume',action='store_true')
    a=ap.parse_args();cfg=yaml.safe_load(Path(a.config).read_text());enforce_gpu_scope()
    if cfg.get('deterministic',True):
        os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.deterministic=True
    rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1))
    device=torch.device('cuda',int(os.environ.get('LOCAL_RANK',0)))
    torch.cuda.set_device(device);torch.set_num_threads(2)
    if world>1:dist.init_process_group('nccl')
    seed=cfg.get('seed',17);torch.manual_seed(seed);random.seed(seed)
    out=Path(cfg['artifact_dir']);out.mkdir(parents=True,exist_ok=True)
    if not (a.resume or a.verify_resume) and (out/'report.json').exists():
        raise ValueError('Existing completed run: choose a new artifact_dir or explicit resume')
    tiny=cfg.get('tiny',False);dtype=torch.bfloat16
    if tiny:
        qc=Qwen3Config(vocab_size=97,hidden_size=32,intermediate_size=48,num_hidden_layers=3,
            num_attention_heads=4,num_key_value_heads=2,head_dim=8,max_position_embeddings=4096,
            tie_word_embeddings=True,pad_token_id=0,eos_token_id=2)
        original=Qwen3ForCausalLM(qc);model=from_qwen_model(original,**cfg.get("model_overrides",{}));teacher=from_qwen_model(original).to(dtype)
        packs=[pack_sequences([[5,7,10,20],[12,21,35,9]],0,cfg['sequence_length'])]
        dev=packs;data_hash={'fixture':True};tokenizer=None
    else:
        if cfg.get('initial_checkpoint'):
            from .modeling_rmt import RmtForCausalLM
            model=RmtForCausalLM.from_pretrained(cfg['initial_checkpoint'],local_files_only=True,dtype=torch.float32)
            for key,value in cfg.get('model_overrides',{}).items():
                if key in ('max_recurrences','num_experts') and getattr(model.config,key)!=value:
                    raise ValueError('Initial checkpoint structural identity differs')
                setattr(model.config,key,value)
            model.config.num_hidden_layers=model.config.num_recurrences
            model.config.layer_types=["full_attention"]*model.config.num_recurrences
        else:
            model,_=load_qwen_as_rmt(cfg['base_model'],dtype=torch.float32,**cfg.get('model_overrides',{}))
        teacher,_=load_qwen_as_rmt(cfg['base_model'],dtype=dtype)
        tokenizer=AutoTokenizer.from_pretrained(cfg['base_model'],local_files_only=True)
        paths={name:cfg[name+'_data'] for name in ['train','dev']}
        data_hash={name:file_sha(path) for name,path in paths.items()}
        packs=load_packs(paths['train'],cfg['sequence_length'],tokenizer.pad_token_id,seed)
        dev=load_packs(paths['dev'],cfg['sequence_length'],tokenizer.pad_token_id,seed,shuffle=False)
    model.config._attn_implementation=cfg.get('attention','sdpa')
    teacher.config._attn_implementation=cfg.get('attention','sdpa')
    mode=cfg.get('routing_mode','layer_order');model.config.routing_mode=mode
    if not cfg.get('initial_checkpoint'):
        with torch.no_grad():model.model.cell.router.weight.normal_(std=cfg.get('router_init_std',0.01))
    teacher=teacher.to(device).eval().requires_grad_(False);model=model.to(device).train()
    if cfg.get('checkpoint',True):model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    if cfg.get('compile',False):model.model.cell.bank.compile_projections(training=True)
    if world>1:
        fully_shard(model,reshard_after_forward=False,
            mp_policy=MixedPrecisionPolicy(param_dtype=dtype,reduce_dtype=torch.float32))
    groups=[{'params':[p for n,p in model.named_parameters() if '.router.' not in n],'lr':cfg.get('learning_rate',1e-5)},
            {'params':[p for n,p in model.named_parameters() if '.router.' in n],'lr':cfg.get('router_learning_rate',1e-4)}]
    opt=torch.optim.AdamW(groups,weight_decay=0.0,foreach=False)
    total_steps=cfg['steps'];warmup=cfg.get('lr_warmup_steps',4)
    scheduler=torch.optim.lr_scheduler.LambdaLR(opt,lambda step:min(1.0,(step+1)/max(1,warmup)))
    code={str(p):file_sha(p) for p in Path('src/rmt').rglob('*.py') if 'evaluation' not in p.parts and 'inference' not in p.parts}
    identity={'config':cfg,'data':data_hash,'world_size':world,'code':code,'environment':versions()}
    # Device ordinal is irrelevant to resume on the same architecture/runtime.
    identity['environment'].pop('cuda_visible_devices',None)
    identity['environment']['gpu']=torch.cuda.get_device_name(device)
    started=time.monotonic();rows=[];checks=[];cursor=0

    def reduce_sum(x):
        if world>1:dist.all_reduce(x)
        return x

    def emit(row):
        if rank==0:
            print(json.dumps(row),flush=True)
            with (out/'events.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')

    def forward(batch,trace=True,step=None):
        teacher_hidden=None
        if cfg.get('kd_weight',0.5):
            with torch.no_grad():
                teacher_hidden=teacher(**{k:v for k,v in batch.items() if k!='labels'},use_cache=False,
                    return_hidden_only=True).last_hidden_state
        with torch.autocast('cuda',dtype=dtype):
            return model(**batch,use_cache=False,loss_chunk_size=cfg.get('loss_chunk_size',64),
                teacher_hidden_states=teacher_hidden,teacher_head_weight=teacher.lm_head.weight,
                kd_weight=cfg.get('kd_weight',0.5),output_router_trace=trace,**recurrence_controls(cfg,step))

    def route_stats(result,batch):
        valid=batch['attention_mask'].bool();counts=torch.zeros(model.config.num_experts,device=device,dtype=torch.long)
        off=torch.zeros((),device=device,dtype=torch.long);off_prior=torch.zeros_like(off)
        for i,route in enumerate(result.router_indices):
            selected=valid & (route>=0)
            counts+=torch.bincount(route[selected],minlength=counts.numel())
            off+=(route[selected]!=i%model.config.num_experts).sum()
            off_prior+=(route[selected]!=prior_expert(model.config,i)).sum()
        reduce_sum(counts);reduce_sum(off);reduce_sum(off_prior)
        depths=torch.bincount(result.exit_depths[valid],minlength=model.config.max_recurrences+1)
        reduce_sum(depths)
        mean=(depths*torch.arange(depths.numel(),device=device)).sum()/depths.sum().clamp_min(1)
        return {'off_layer_fraction':(off/counts.sum().clamp_min(1)).item(),'expert_counts':counts.tolist(),
                'off_prior_fraction':(off_prior/counts.sum().clamp_min(1)).item(),
                'exit_depth_counts':depths.tolist(),'mean_recurrences':mean.item()}

    def train_step(step):
        model.train();prior=prior_at(cfg,step);model.model.cell.router.prior_strength=prior
        batch=batch_at(packs,step,rank,world,device);opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize();begin=time.monotonic();torch.cuda.reset_peak_memory_stats()
        result=forward(batch,step=step)
        nt=(shifted_targets(batch['labels'],batch['attention_mask'],batch['segment_ids'])!=-100).sum()
        totals=reduce_sum(torch.stack((nt,batch['attention_mask'].sum())))
        if totals[0].item()==0:raise ValueError('No supervised targets')
        (result.loss*(world*nt/totals[0])).backward()
        squared=torch.zeros((),device=device)
        for p in model.parameters():
            if p.grad is not None:squared+=local(p.grad).float().square().sum()
        # FSDP2 shards collectively represent one model; norm is global.
        if world>1:reduce_sum(squared)
        norm=squared.sqrt()
        if not torch.isfinite(norm):raise FloatingPointError('Nonfinite global gradient norm')
        scale=(cfg.get('max_grad_norm',1.0)/(norm+1e-6)).clamp(max=1)
        for p in model.parameters():
            if p.grad is not None:local(p.grad).mul_(scale)
        opt.step();scheduler.step();torch.cuda.synchronize()
        seconds=torch.tensor(time.monotonic()-begin,device=device)
        if world>1:dist.all_reduce(seconds,op=dist.ReduceOp.MAX)
        stats=route_stats(result,batch)
        loss_values=reduce_sum(torch.stack((result.loss.detach(),result.ce_loss.detach(),result.kd_loss.detach()))*nt)/totals[0]
        peak=torch.tensor(torch.cuda.max_memory_allocated(),device=device,dtype=torch.long)
        if world>1:dist.all_reduce(peak,op=dist.ReduceOp.MAX)
        row={'event':'train','step':step,'prior':prior,'loss':loss_values[0].item(),'ce':loss_values[1].item(),
            'kl':loss_values[2].item(),'grad_norm':norm.item(),'seconds':seconds.item(),
            'input_tokens':totals[1].item(),'targets':totals[0].item(),
            'input_tokens_per_second':(totals[1]/seconds).item(),'targets_per_second':(totals[0]/seconds).item(),
            'peak_allocated_bytes':peak.item(),**stats}
        emit(row);return row

    @torch.no_grad()
    def validate(step,prior=None):
        model.eval();saved=model.model.cell.router.prior_strength
        model.model.cell.router.prior_strength=prior_at(cfg,step) if prior is None else prior
        sums=torch.zeros(4,device=device);off=torch.zeros((),device=device);used=torch.zeros((),device=device)
        # Every rank executes the same number of forwards, for FSDP collectives.
        rounds=cfg.get('validation_batches',2)
        for i in range(rounds):
            batch=batch_at(dev,i,rank,world,device)
            include=i*world+rank<len(dev)
            if not include:batch['labels']=torch.full_like(batch['labels'],-100)
            result=forward(batch)
            nt=(shifted_targets(batch['labels'],batch['attention_mask'],batch['segment_ids'])!=-100).sum()
            sums+=torch.stack((result.ce_loss*nt,result.kd_loss*nt,nt.float(),batch['attention_mask'].sum().float()))
            valid=batch['attention_mask'].bool() & include
            for j,route in enumerate(result.router_indices):
                selected=valid & (route>=0)
                off+=(route[selected]!=j%model.config.num_experts).sum();used+=selected.sum()
        reduce_sum(sums);reduce_sum(off);reduce_sum(used)
        row={'event':'validation','step':step,'prior':model.model.cell.router.prior_strength,'ce':(sums[0]/sums[2]).item(),
             'kl':(sums[1]/sums[2]).item(),'targets':sums[2].item(),'off_layer_fraction':(off/used.clamp_min(1)).item(),'unique_packs':min(rounds*world,len(dev))}
        if not torch.isfinite(sums).all():raise FloatingPointError('Nonfinite validation')
        model.model.cell.router.prior_strength=saved;model.train();emit(row);return row

    def save(step):
        options=StateDictOptions(cpu_offload=True)
        m,o=get_state_dict(model,opt,options=options)
        dcp.save({'model':m,'optimizer':o},checkpoint_id=out/'resume')
        runtime={'identity':identity,'cursor':step,'scheduler':scheduler.state_dict(),
                 'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state(),
                 'python_rng':random.getstate(),'prior':model.model.cell.router.prior_strength,'rows':rows,'checks':checks}
        torch.save(runtime,out/f'rank_{rank}_runtime.pt')
        if world>1:dist.barrier()

    def restore():
        runtime=torch.load(out/f'rank_{rank}_runtime.pt',weights_only=False)
        if runtime['identity']!=identity:raise ValueError('Resume identity changed: config/data/world/code/runtime must match')
        options=StateDictOptions(cpu_offload=True);m,o=get_state_dict(model,opt,options=options)
        state={'model':m,'optimizer':o};dcp.load(state,checkpoint_id=out/'resume')
        set_state_dict(model,opt,model_state_dict=state['model'],optim_state_dict=state['optimizer'],options=options)
        scheduler.load_state_dict(runtime['scheduler']);torch.set_rng_state(runtime['torch_rng'])
        torch.cuda.set_rng_state(runtime['cuda_rng']);random.setstate(runtime['python_rng'])
        model.model.cell.router.prior_strength=runtime['prior']
        return runtime

    if a.resume or a.verify_resume:
        runtime=restore();cursor=runtime['cursor'];rows=runtime['rows'];checks=runtime['checks']
    if a.verify_resume:
        reference=json.loads((out/f'rank_{rank}_replay.json').read_text());row=train_step(cursor)
        assert row['loss']==reference['loss'] and parameter_digest(model)==reference['parameter_sha256']
        if rank==0:write_report(out/'fresh_resume.json',{'status':'passed','world_size':world,'next_loss':row['loss'],'parameter_sha256_equal':True})
        if world>1:dist.destroy_process_group()
        return
    if cursor==0 and cfg.get('validation_batches',2):checks.append(validate(0))
    for step in range(cursor,total_steps):
        rows.append(train_step(step))
        if cfg.get('eval_every',0) and (step+1)%cfg['eval_every']==0:checks.append(validate(step+1))
        if cfg.get('save_every',0) and (step+1)%cfg['save_every']==0:save(step+1)
    if cfg.get('validation_batches',2):checks.append(validate(total_steps))
    if cfg.get('save',True):save(total_steps)
    # HF export gathers the final model only; optimizer checkpoint remains sharded.
    if cfg.get('export',True):
        opts=StateDictOptions(full_state_dict=True,cpu_offload=True)
        state,_=get_state_dict(model,[],options=opts)
        if rank==0:
            from .modeling_rmt import RmtForCausalLM
            from .configuration_rmt import RmtConfig
            config=RmtConfig.from_dict(model.config.to_dict());config.router_prior_strength=prior_at(cfg,total_steps)
            target=RmtForCausalLM(config).to(dtype=dtype)
            target.load_state_dict(state);target.tie_weights();target.eval()
            export_hf(target,out/'hf',None if tiny else cfg['base_model'],generation_config=teacher.generation_config)
            del target
        del state
        if world>1:dist.barrier()
    if cfg.get('verify_replay',True):
        if not cfg.get('save',True):raise ValueError('Replay verification requires a saved checkpoint')
        reference=train_step(total_steps);sha=parameter_digest(model)
        write_report(out/f'rank_{rank}_replay.json',{'loss':reference['loss'],'parameter_sha256':sha})
        restore();replayed=train_step(total_steps)
        assert reference['loss']==replayed['loss'] and sha==parameter_digest(model)
    if rank==0:
        from torch._dynamo.utils import counters
        report={'status':'passed','identity':identity,'steps':rows,'validation':checks,
                'wall_seconds':time.monotonic()-started,'train_packs':len(packs),'dev_packs':len(dev),
                'compile_counters':{str(k):dict(v) for k,v in counters.items()},
                'scope':'Short instruction-training experiment, not a quality superiority claim'}
        write_report(out/'report.json',report);write_report(cfg['report'],report)
    if world>1:dist.destroy_process_group()


if __name__=='__main__':main()

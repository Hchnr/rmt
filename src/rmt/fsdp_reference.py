"""Two-rank FSDP2 loss/gradient/update versus the identical global batch."""
import copy
import os
import torch
import torch.distributed as dist
from torch.distributed.fsdp import fully_shard
from transformers import Qwen3Config, Qwen3ForCausalLM
from .checkpoint import from_qwen_model
from .data import pack_sequences
from .runtime import enforce_gpu_scope, write_report


def main():
    os.environ["NVIDIA_TF32_OVERRIDE"]="0"
    os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"]="0"
    enforce_gpu_scope()
    rank=int(os.environ['RANK']); world=int(os.environ['WORLD_SIZE'])
    device=torch.device('cuda',int(os.environ['LOCAL_RANK']))
    torch.cuda.set_device(device); torch.set_num_threads(2); dist.init_process_group('nccl')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    torch.manual_seed(31)
    cfg=Qwen3Config(vocab_size=97,hidden_size=32,intermediate_size=48,num_hidden_layers=3,
        num_attention_heads=4,num_key_value_heads=2,head_dim=8,tie_word_embeddings=True,pad_token_id=0)
    cfg._attn_implementation='eager'
    model=from_qwen_model(Qwen3ForCausalLM(cfg)).to(device).train()
    model.model.cell.router.prior_strength=0
    with torch.no_grad(): model.model.cell.router.weight.normal_(std=.03)
    reference=copy.deepcopy(model)
    fully_shard(model,reshard_after_forward=False)
    optimizer=torch.optim.SGD(model.parameters(),lr=.05)
    ref_optimizer=torch.optim.SGD(reference.parameters(),lr=.05)
    batches=[pack_sequences([[5+i,7,10],[12,20+i,25]],length=8,device=device) for i in range(world)]
    global_batch={key:torch.cat([b[key] for b in batches],dim=0) for key in batches[0]}
    results=[]
    for mode in ['forced','learned']:
        optimizer.zero_grad(set_to_none=True); ref_optimizer.zero_grad(set_to_none=True)
        routes=[torch.full((1,8,3),i%2,device=device,dtype=torch.long) for i in range(world)]
        extra={} if mode=='learned' else {'forced_routes':routes[rank]}
        ref_extra={} if mode=='learned' else {'forced_routes':torch.cat(routes,dim=0)}
        output=model(**batches[rank],use_cache=False,routing_mode=mode,**extra)
        expected=reference(**global_batch,use_cache=False,routing_mode=mode,**ref_extra)
        mean_loss=output.loss.detach().clone(); dist.all_reduce(mean_loss); mean_loss/=world
        torch.testing.assert_close(mean_loss,expected.loss,atol=1e-6,rtol=1e-5)
        output.loss.backward(); expected.loss.backward()
        max_grad=0.
        ref_params=dict(reference.named_parameters())
        for name,p in model.named_parameters():
            q=ref_params[name]
            if p.grad is None:
                assert q.grad is None or torch.count_nonzero(q.grad)==0
                continue
            actual=p.grad.full_tensor()
            wanted=torch.zeros_like(q) if q.grad is None else q.grad
            max_grad=max(max_grad,(actual-wanted).abs().max().item())
            torch.testing.assert_close(actual,wanted,atol=2e-6,rtol=2e-4,msg=lambda message: name+' '+message)
        optimizer.step(); ref_optimizer.step()
        max_update=0.
        for name,p in model.named_parameters():
            actual=p.full_tensor(); wanted=ref_params[name]
            max_update=max(max_update,(actual-wanted).abs().max().item())
            torch.testing.assert_close(actual,wanted,atol=2e-6,rtol=2e-4,msg=lambda message: name+' '+message)
        results.append({'mode':mode,'loss':mean_loss.item(),'max_gradient_difference':max_grad,
                        'max_parameter_difference':max_update})
    if rank==0: write_report('reports/bootstrap/fsdp_reference.json',{'status':'passed','world_size':world,'checks':results})
    dist.destroy_process_group()


if __name__=='__main__': main()

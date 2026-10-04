"""Stress collective symmetry with empty ranks and unrelated per-token exit paths."""
import argparse
import copy
import json
import os
from pathlib import Path
import torch
import torch.distributed as dist
from torch.distributed.fsdp import fully_shard
from rmt.configuration_rmt import RmtConfig
from rmt.modeling_rmt import RmtForCausalLM
p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--pattern-offset',type=int,default=0);a=p.parse_args()
rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
dist.init_process_group('nccl');torch.manual_seed(42);torch.set_num_threads(2)
c=RmtConfig(vocab_size=97,hidden_size=32,intermediate_size=48,num_hidden_layers=3,num_experts=3,
    num_recurrences=3,max_recurrences=6,min_recurrences=2,halting_policy='hidden',
    num_attention_heads=4,num_key_value_heads=2,head_dim=8,pad_token_id=0,tie_word_embeddings=True)
c._attn_implementation='eager';model=RmtForCausalLM(c).cuda().train();reference=copy.deepcopy(model)
model.model.gradient_checkpointing=True;fully_shard(model,reshard_after_forward=False)
def batch(r):
 r+=a.pattern_offset
 ids=torch.tensor([[4,7,9,11]],device='cuda');mask=torch.ones_like(ids);labels=ids.clone()
 if r%3==0:mask.zero_();labels.fill_(-100)
 stops=torch.tensor([[2,6,3,5] if r%3==1 else [6,6,6,6]],device='cuda')
 return dict(input_ids=ids,attention_mask=mask,labels=labels,forced_exit_depths=stops,use_cache=False,loss_chunk_size=2)
out=model(**batch(rank));out.loss.backward()
for r in range(world):(reference(**batch(r)).loss/world).backward()
refs=dict(reference.named_parameters());maximum=0.;compared=0
for name,param in model.named_parameters():
 presence=torch.tensor(int(param.grad is not None),device='cuda');dist.all_reduce(presence)
 if presence.item() not in [0,world]:raise AssertionError('Rank-dependent gradient presence: '+name)
 if param.grad is None:
  assert refs[name].grad is None or refs[name].grad.abs().max()==0
  continue
 actual=param.grad.full_tensor() if hasattr(param.grad,'full_tensor') else param.grad
 expected=refs[name].grad
 if expected is None:expected=torch.zeros_like(actual)
 error=(actual-expected).abs().max().item();maximum=max(maximum,error);compared+=1
 torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-5)
record={'rank':rank,'loss':out.loss.item(),'depths':out.exit_depths.tolist(),'max_gradient_abs_error':maximum,'compared_parameters':compared}
records=[None]*world;dist.all_gather_object(records,record)
if rank==0:
 report={'status':'passed','world_size':world,'pattern_offset':a.pattern_offset,'checks':'FSDP2 + recomputation, empty ranks, mixed/capped token paths; gradients against independent unsharded rank-average reference','ranks':records}
 Path(a.output).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
dist.destroy_process_group()

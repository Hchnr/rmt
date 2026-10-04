"""Diagnostic profiler ranges in an isolated process; never changes saved models."""
import argparse
from functools import wraps
import json
from pathlib import Path
import time
import torch
from torch.profiler import profile, ProfilerActivity, record_function
from rmt.inference.runner import Runner, Generation
import rmt.modeling_rmt as modeling
from rmt.experts import BoundExpertBank
from rmt.routing import BoundRouter
from rmt.cache import RmtCapacityCache

p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--output',required=True)
a=p.parse_args()
r=Runner(a.model,compiled=True,attention='sdpa',max_context=4096)
prompts=[r.render([{'role':'user','content':f'Explain in detail how to test a sorting algorithm. Include example {i+1}.'}]) for i in range(4)]
configs=[Generation(max_new_tokens=8,temperature=0,presence_penalty=0) for _ in prompts]
reference=r.generate(prompts,configs)
originals=[]
def instrument(owner,name,label):
 original=getattr(owner,name);originals.append((owner,name,original))
 @wraps(original)
 def measured(*args,**kwargs):
  with record_function('rmt::'+label):return original(*args,**kwargs)
 setattr(owner,name,measured)
for owner,name,label in [(modeling,'attend','attention'),(modeling,'hidden_stable','hidden_check'),
 (modeling,'probability_stable','probability_check'),(BoundExpertBank,'qkv','qkv_dispatch_projection'),
 (BoundExpertBank,'output','output_ffn_dispatch_projection'),(BoundExpertBank,'groups','expert_groups'),
 (BoundRouter,'forward','router'),(RmtCapacityCache,'update','kv_update')]:instrument(owner,name,label)
start=time.monotonic()
try:
 with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA],record_shapes=False,profile_memory=False,with_stack=False) as prof:
  output=r.generate(prompts,configs)
finally:
 for owner,name,original in originals:setattr(owner,name,original)
assert [x['token_ids'] for x in output]==[x['token_ids'] for x in reference]
keys=prof.key_averages()
def row(x):return {'name':x.key,'calls':x.count,'cpu_total_us':x.cpu_time_total,
 'self_cpu_us':x.self_cpu_time_total,'device_total_us':x.device_time_total,'self_device_us':x.self_device_time_total}
result={'status':'passed','model':a.model,'batch_size':4,'token_budget':8,
 'profiled_wall_seconds':time.monotonic()-start,'tokens_unchanged':True,
 'regions':[row(x) for x in keys if x.key.startswith('rmt::')],
 'top_self_cpu':sorted([row(x) for x in keys],key=lambda x:x['self_cpu_us'],reverse=True)[:20],
 'top_self_device':sorted([row(x) for x in keys],key=lambda x:x['self_device_us'],reverse=True)[:20],
 'scope':'One warmed prefill plus seven cached decode steps, greedy batch4, isolated process on shared machine. Profiler overhead is substantial; inclusive nested regions must not be summed. Device ranges may omit CUDA-graph internals. Diagnostic only, not latency benchmark or explanation of all 32768-token behavior.'}
Path(a.output).write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'status':result['status'],'regions':result['regions']}),flush=True)

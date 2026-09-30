"""Detect depth-phase plateaus on a fixed full unroll, without oracle stopping."""
import argparse
import json
from pathlib import Path
import torch
from rmt.modeling_rmt import RmtForCausalLM
from rmt.halting import hidden_stable,prior_expert
p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--output',required=True);a=p.parse_args()
torch.set_num_threads(4);model=RmtForCausalLM.from_pretrained(a.model,dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).cuda().eval()
rows=[json.loads(x) for x in Path('artifacts/v0.0.4_dynamic_recurr/corpus/halt_calibration.jsonl').read_text().splitlines()][:16]
values={depth:{'tokens':0,'stable_twice':0,'plateau_then_unstable':0,'lags':{str(lag):[] for lag in [1,2,8]}} for depth in range(36,48)}
with torch.inference_mode():
 for row in rows:
  raw=[];hook=model.model.norm.register_forward_pre_hook(lambda module,args:raw.append(args[0].clone()))
  out=model(torch.tensor([row['input_ids'][:256]],device='cuda'),halting_policy='fixed',recurrence_limit=48,output_hidden_states=True,use_cache=False,logits_to_keep=1)
  hook.remove();states=list(out.hidden_states);states[-1]=raw[-1]
  stable=[None]+[hidden_stable(states[i-1],states[i],model.config) for i in range(1,49)]
  for depth,v in values.items():
   twice=stable[depth]&stable[depth-1];v['tokens']+=twice.numel();v['stable_twice']+=twice.sum().item()
   v['plateau_then_unstable']+=(twice & ~stable[depth+1]).sum().item()
   for lag in [1,2,8]:
    previous=states[depth-lag].float();current=states[depth].float()
    relative=(current-previous).square().mean(-1).sqrt()/previous.square().mean(-1).sqrt().clamp_min(1e-6)
    v['lags'][str(lag)].extend(relative.flatten().cpu().tolist())
for depth,v in values.items():
 v['prior_expert']=prior_expert(model.config,depth-1)
 v['relative_update_quantiles']={lag:torch.quantile(torch.tensor(x),torch.tensor([.1,.5,.9])).tolist() for lag,x in v.pop('lags').items()}
 v['stable_twice_fraction']=v['stable_twice']/v['tokens']
 v['next_step_unstable_given_stable_twice']=v['plateau_then_unstable']/max(1,v['stable_twice'])
r={'model':a.model,'config':model.config.to_dict(),'selection_ids':[x['id'] for x in rows],'prefix_limit':256,'depths':values,
 'scope':'16 calibration prefixes, same-weight fixed 48 unroll, lag-1/2/8 raw hidden updates; context KV differs from actual early-exit execution. Plateau evidence is not answer correctness.'}
Path(a.output).write_text(json.dumps(r,indent=2)+'\n')

"""Same-weight fixed-depth counterfactuals; token CE evidence, never oracle task scores."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from rmt.modeling_rmt import RmtForCausalLM
p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--output',required=True)
p.add_argument('--data',default='artifacts/v0.0.4_dynamic_recurr/corpus/dev.jsonl');p.add_argument('--limit',type=int,default=64)
a=p.parse_args();torch.set_num_threads(4)
m=RmtForCausalLM.from_pretrained(a.model,dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).cuda().eval()
rows=[json.loads(x) for x in Path(a.data).read_text().splitlines()][:a.limit];results=[]
with torch.inference_mode():
 for row in rows:
  ids=torch.tensor([row['input_ids']],device='cuda');labels=torch.tensor(row['labels'][1:],device='cuda');valid=labels!=-100
  paths={}
  for depth in [36,40,48,None]:
   result=m.model(ids,use_cache=False,halting_policy='fixed' if depth else None,recurrence_limit=depth,head_weight=m.lm_head.weight)
   hidden=result[0][0,:-1][valid];targets=labels[valid];loss=[];correct=[]
   for start in range(0,len(targets),64):
    logits=F.linear(hidden[start:start+64],m.lm_head.weight).float()
    loss.append(F.cross_entropy(logits,targets[start:start+64],reduction='none'));correct.append(logits.argmax(-1)==targets[start:start+64])
   paths[str(depth)]={'loss':torch.cat(loss),'correct':torch.cat(correct),'depth':result[5][0,:-1][valid]}
  d=paths['None'];last=paths['48'];early=d['depth']<48
  results.append({'id':row['id'],'targets':len(targets),'ce':{k:v['loss'].mean().item() for k,v in paths.items()},
   'mean_depth':d['depth'].float().mean().item(),'early_targets':early.sum().item(),
   'early_nll_improved_by_fixed48':(early & (last['loss']<d['loss']-.01)).sum().item(),
   'early_nll_harmed_by_fixed48':(early & (last['loss']>d['loss']+.01)).sum().item(),
   'early_top1_mismatch':(early & ~d['correct']).sum().item(),
   'early_top1_mismatch_fixed48_correct':(early & ~d['correct'] & last['correct']).sum().item()})
r={'model':a.model,'data_sha256':hashlib.sha256(Path(a.data).read_bytes()).hexdigest(),'config':m.config.to_dict(),
 'documents':len(results),'targets':sum(x['targets'] for x in results),'rows':results,
 'scope':'Teacher-forced next-token CE/top1 only; fixed-depth paths also change context KV. Not answer correctness, oracle deployment, or a causally isolated extra-step treatment.'}
r['ce']={k:sum(x['ce'][k]*x['targets'] for x in results)/r['targets'] for k in ['36','40','48','None']}
r['counts']={k:sum(x[k] for x in results) for k in results[0] if k.startswith('early_')}
Path(a.output).write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({'ce':r['ce'],'counts':r['counts']}))

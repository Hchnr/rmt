"""Measure deployment depths on frozen calibration prompts, without scoring answers."""
import argparse
import hashlib
import json
from pathlib import Path
import time
from rmt.inference.runner import Runner,Generation
p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--output',required=True)
p.add_argument('--data',default='artifacts/v0.0.4_dynamic_recurr/corpus/halt_calibration.jsonl');p.add_argument('--per-domain',type=int,default=8)
p.add_argument('--max-new-tokens',type=int,default=128);a=p.parse_args()
r=Runner(a.model,compiled=True,attention='sdpa',max_context=4096);selected=[];counts={}
for row in map(json.loads,Path(a.data).read_text().splitlines()):
 domain=row['domain']
 if counts.get(domain,0)>=a.per_domain:continue
 prompt=r.render(row['messages'][:1])
 if len(r.tokenizer.encode(prompt,add_special_tokens=False))>1024:continue
 counts[domain]=counts.get(domain,0)+1;selected.append((row,prompt))
outputs=[];start=time.monotonic()
for i in range(0,len(selected),4):
 batch=selected[i:i+4];values=r.generate([x[1] for x in batch],[Generation(max_new_tokens=a.max_new_tokens,temperature=0,presence_penalty=0) for _ in batch])
 for (source,prompt),value in zip(batch,values):outputs.append({'id':source['id'],'domain':source['domain'],**value})
 print(json.dumps({'completed':len(outputs),'seconds':time.monotonic()-start}),flush=True)
totals={k:sum(x['recurrence'][k] for x in outputs) for k in outputs[0]['recurrence']}
report={'model':a.model,'config':r.model.config.to_dict(),'data_sha256':hashlib.sha256(Path(a.data).read_bytes()).hexdigest(),
 'records':len(outputs),'max_new_tokens':a.max_new_tokens,'domain_counts':counts,'totals':totals,
 'prefill_mean_depth':totals['prefill_depth_sum']/totals['prefill_positions'],
 'decode_mean_depth':totals['decode_depth_sum']/max(1,totals['decode_positions']),
 'wall_seconds':time.monotonic()-start,'outputs':outputs,
 'scope':'Calibration-only depth/cost sample, greedy capped output; no answer quality metric. Final generated token has no subsequent decode forward.'}
Path(a.output).write_text(json.dumps(report,indent=2)+'\n')

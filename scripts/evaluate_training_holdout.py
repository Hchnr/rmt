"""Evaluate every unique held-out corpus pack, with assistant targets only."""
import argparse
import json
from pathlib import Path
import time
import torch
from transformers import AutoTokenizer
from rmt.checkpoint import load_qwen_as_rmt
from rmt.configuration_rmt import RmtConfig
from rmt.modeling_rmt import RmtForCausalLM
from rmt.training_data import load_packs,file_sha
from rmt.losses import shifted_targets
p=argparse.ArgumentParser();p.add_argument('--model');p.add_argument('--name',required=True);p.add_argument('--routing',choices=['layer_order','learned']);a=p.parse_args()
torch.set_num_threads(2);torch.manual_seed(17)
base='/share/project/eai_pwm/models/Qwen/Qwen3-4B'
if a.model:
 config=RmtConfig.from_pretrained(a.model);config._attn_implementation='sdpa'
 if a.routing:config.routing_mode=a.routing
 model=RmtForCausalLM.from_pretrained(a.model,config=config,dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval()
else:
 model,_=load_qwen_as_rmt(base,dtype=torch.bfloat16);model.config._attn_implementation='sdpa';model=model.cuda().eval()
tokenizer=AutoTokenizer.from_pretrained(base,local_files_only=True)
path='artifacts/v0.0.4/corpus/dev.jsonl';packs=load_packs(path,2048,tokenizer.pad_token_id,shuffle=False)
total=0.;targets=0;off=0;decisions=0;start=time.monotonic()
with torch.inference_mode():
 for i,batch in enumerate(packs):
  batch={k:v.cuda() for k,v in batch.items()};result=model(**batch,use_cache=False,loss_chunk_size=64,output_router_trace=True)
  n=int((shifted_targets(batch['labels'],batch['attention_mask'],batch['segment_ids'])!=-100).sum())
  total+=result.ce_loss.item()*n;targets+=n;valid=batch['attention_mask'].bool()
  for step,chosen in enumerate(result.router_indices):off+=(chosen[valid]!=step).sum().item();decisions+=valid.sum().item()
  if (i+1)%10==0:print(json.dumps({'packs':i+1,'ce':total/targets}),flush=True)
value={'status':'passed','name':a.name,'model':a.model or base,'routing_mode':model.config.routing_mode,
 'prior':model.model.cell.router.prior_strength,'ce':total/targets,'targets':targets,'unique_packs':len(packs),
 'records':sum(1 for _ in Path(path).open()),'off_layer_fraction':off/decisions,'seconds':time.monotonic()-start,
 'data_sha256':file_sha(path),'scope':'Entire 500-record English instruction holdout; no repeated packs; no claim of benchmark superiority'}
Path(f'reports/v0.0.4/holdout_{a.name}.json').write_text(json.dumps(value,indent=2)+'\n');print(value)

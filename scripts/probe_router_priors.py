"""Held-out sensitivity to prior strength; no optimizer updates or benchmark use."""
import argparse
import json
from pathlib import Path
import torch
from transformers import AutoTokenizer
from rmt.checkpoint import load_qwen_as_rmt
from rmt.training_data import load_packs
from rmt.losses import shifted_targets
p=argparse.ArgumentParser();p.add_argument('--output',default='reports/v0.0.4/prior_sensitivity.json');a=p.parse_args()
torch.set_num_threads(2);torch.manual_seed(17)
base='/share/project/eai_pwm/models/Qwen/Qwen3-4B'
model,_=load_qwen_as_rmt(base,dtype=torch.float32);model=model.to(dtype=torch.bfloat16,device='cuda').eval()
model.model.cell.router.float()
with torch.no_grad():model.model.cell.router.weight.normal_(std=0.01)
model.config._attn_implementation='sdpa';tokenizer=AutoTokenizer.from_pretrained(base,local_files_only=True)
packs=load_packs('artifacts/v0.0.4/corpus/dev_pilot.jsonl',512,tokenizer.pad_token_id,shuffle=False)[:4]
rows=[]
with torch.inference_mode():
 for prior in [4.0,3.5,3.0,2.5,2.0,1.0,0.0]:
  model.model.cell.router.prior_strength=prior;total=0;count=0;off=0;routes=0
  for batch in packs:
   batch={k:v.cuda() for k,v in batch.items()};result=model(**batch,use_cache=False,loss_chunk_size=64,output_router_trace=True,routing_mode='learned')
   n=(shifted_targets(batch['labels'],batch['attention_mask'],batch['segment_ids'])!=-100).sum().item()
   total+=result.ce_loss.item()*n;count+=n;valid=batch['attention_mask'].bool()
   for i,x in enumerate(result.router_indices):off+=(x[valid]!=i).sum().item();routes+=valid.sum().item()
  row={'prior':prior,'ce':total/count,'targets':count,'off_layer_fraction':off/routes};rows.append(row);print(row,flush=True)
Path(a.output).write_text(json.dumps({'scope':'Initial random router, fixed first four held-out packs; diagnostic only, no optimization','rows':rows},indent=2)+'\n')

"""Check trained HF reload, routing prior, and held-out loss against training."""
import argparse
import json
from pathlib import Path
import torch
from transformers import AutoTokenizer
from rmt.configuration_rmt import RmtConfig
from rmt.modeling_rmt import RmtForCausalLM
from rmt.training_data import load_packs,batch_at
from rmt.losses import shifted_targets
from rmt.data import pack_sequences
from rmt.train import prior_at
p=argparse.ArgumentParser();p.add_argument('work');a=p.parse_args();work=Path(a.work)
report=json.loads((work/'report.json').read_text());cfg=report['identity']['config'];world=report['identity']['world_size']
config=RmtConfig.from_pretrained(work/'hf');config._attn_implementation=cfg.get('attention','sdpa')
model=RmtForCausalLM.from_pretrained(work/'hf',config=config,dtype=torch.bfloat16,attn_implementation=config._attn_implementation).cuda().eval()
assert config.routing_mode==cfg['routing_mode'];assert config.router_prior_strength==prior_at(cfg,report.get('completed_steps',cfg['steps']))
if cfg.get('tiny'):
 packs=[pack_sequences([[5,7,10,20],[12,21,35,9]],0,cfg['sequence_length'])]
else:
 tokenizer=AutoTokenizer.from_pretrained(work/'hf',local_files_only=True)
 packs=load_packs(cfg['dev_data'],cfg['sequence_length'],tokenizer.pad_token_id,cfg['seed'],shuffle=False)
total=0.;count=0
with torch.inference_mode():
 for step in range(cfg.get('validation_batches',2)):
  for rank in range(world):
   if step*world+rank>=len(packs):continue
   batch=batch_at(packs,step,rank,world,'cuda');result=model(**batch,use_cache=False,loss_chunk_size=cfg.get('loss_chunk_size',64))
   n=(shifted_targets(batch['labels'],batch['attention_mask'],batch['segment_ids'])!=-100).sum().item();total+=result.ce_loss.item()*n;count+=n
expected=report['validation'][-1]['ce'];observed=total/count
assert abs(expected-observed)<0.005,(expected,observed)
value={'status':'passed','work':str(work),'expected_training_dev_ce':expected,'hf_reload_dev_ce':observed,
 'absolute_difference':abs(expected-observed),'tolerance':0.005,'routing_mode':config.routing_mode,'prior':config.router_prior_strength,
 'note':'BF16 export of FP32 master weights; validation checks observed loss, not universal bitwise identity'}
(work/'hf_reload.json').write_text(json.dumps(value,indent=2)+'\n');print(value)

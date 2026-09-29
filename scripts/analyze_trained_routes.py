"""Locate route changes in the frozen candidate; descriptive, no checkpoint selection."""
import json
from pathlib import Path
import torch
from transformers import AutoTokenizer
from rmt.configuration_rmt import RmtConfig
from rmt.modeling_rmt import RmtForCausalLM
from rmt.training_data import load_packs
from rmt.losses import shifted_targets
torch.set_num_threads(2)
path='artifacts/v0.0.4/train/opened_128_deterministic/hf'
config=RmtConfig.from_pretrained(path);config._attn_implementation='sdpa'
model=RmtForCausalLM.from_pretrained(path,config=config,dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval()
tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True)
packs=load_packs('artifacts/v0.0.4/corpus/dev.jsonl',2048,tokenizer.pad_token_id,shuffle=False)
matrix=torch.zeros(config.num_recurrences,config.num_experts,device='cuda',dtype=torch.long)
tokens=changed_tokens=targets=changed_targets=multi=0
with torch.inference_mode():
 for batch in packs:
  batch={k:v.cuda() for k,v in batch.items()};result=model(**{k:v for k,v in batch.items() if k!='labels'},use_cache=False,return_hidden_only=True,output_router_trace=True)
  routes=torch.stack(result.router_indices);changed=(routes!=torch.arange(config.num_recurrences,device='cuda')[:,None,None]);valid=batch['attention_mask'].bool()
  target_mask=torch.cat((shifted_targets(batch['labels'],batch['attention_mask'],batch['segment_ids'])!=-100,torch.zeros_like(valid[:,:1])),dim=1)
  tokens+=valid.sum().item();changed_tokens+=(changed.any(0)&valid).sum().item();multi+=((changed.sum(0)>1)&valid).sum().item()
  targets+=target_mask.sum().item();changed_targets+=(changed.any(0)&target_mask).sum().item()
  for step,chosen in enumerate(routes):matrix[step]+=torch.bincount(chosen[valid],minlength=config.num_experts)
counts=matrix.cpu().tolist();off=[{'step':i,'expert':j,'count':counts[i][j]} for i in range(config.num_recurrences) for j in range(config.num_experts) if i!=j and counts[i][j]]
value={'model':path,'scope':'500-record development corpus, all valid input positions; not a causal attribution of downstream errors',
 'input_tokens':tokens,'tokens_with_any_route_change':changed_tokens,'changed_token_fraction':changed_tokens/tokens,
 'tokens_with_multiple_route_changes':multi,'supervised_predictor_positions':targets,
 'changed_supervised_predictor_positions':changed_targets,'changed_supervised_fraction':changed_targets/targets,
 'off_layer_decisions':sum(x['count'] for x in off),'off_layer_fraction':sum(x['count'] for x in off)/(tokens*config.num_recurrences),
 'top_transitions':sorted(off,key=lambda x:-x['count'])[:20],'step_expert_counts':counts}
Path('reports/v0.0.4/trained_route_analysis.json').write_text(json.dumps(value,indent=2)+'\n');print({k:v for k,v in value.items() if k!='step_expert_counts'})

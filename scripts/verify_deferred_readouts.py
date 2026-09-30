"""Real 4B P/H+P parity and measured readout deferral cost."""
import argparse
import json
from pathlib import Path
import statistics
import time
import torch
from rmt.modeling_rmt import RmtForCausalLM
from rmt.cache import RmtCapacityCache
p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--output',required=True);a=p.parse_args()
torch.set_num_threads(4);m=RmtForCausalLM.from_pretrained(a.model,dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).cuda().eval()
row=json.loads(Path('artifacts/v0.0.4_dynamic_recurr/corpus/halt_calibration.jsonl').read_text().splitlines()[0]);ids=torch.tensor([row['input_ids'][:256]],device='cuda');reports=[]
with torch.inference_mode():
 for policy in ['probability','hybrid']:
  m.config.halting_policy=policy;m.config.halt_probability_threshold=.001;m.config.halt_threshold=m.config.halt_relative_threshold=.2
  outputs=[];times=[];decodes=[]
  for deferred in [False,True]:
   m.config.defer_probability_checks=deferred;m(ids,use_cache=False,logits_to_keep=1)
   measurements=[]
   for _ in range(3):
    torch.cuda.synchronize();start=time.monotonic();out=m(ids,use_cache=False,logits_to_keep=1);torch.cuda.synchronize();measurements.append(time.monotonic()-start)
   outputs.append((out.logits.clone(),out.exit_depths.clone()));times.append(statistics.median(measurements))
   cache=RmtCapacityCache(ids.shape[1]+8);m(ids[:,:-3],past_key_values=cache,use_cache=True,logits_to_keep=1)
   last=m(ids[:,-3:],past_key_values=cache,use_cache=True);decodes.append((last.logits.clone(),last.exit_depths.clone()))
  error=(outputs[0][0]-outputs[1][0]).abs().max().item();cache_error=(decodes[0][0]-decodes[1][0]).abs().max().item()
  assert error==0 and cache_error==0
  assert torch.equal(outputs[0][1],outputs[1][1]) and torch.equal(decodes[0][1],decodes[1][1])
  reports.append({'policy':policy,'prefix_tokens':ids.numel(),'max_abs_logit_error':error,'cached_chunk_max_abs_error':cache_error,
    'reference_seconds':times[0],'deferred_seconds':times[1],'speedup':times[0]/times[1]})
result={'status':'passed','model':a.model,'rows':reports,'scope':'One real calibration prefix, BF16 inference; tiny tests separately verify exact gradients. Not a universal bitwise or generation throughput claim.'}
Path(a.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

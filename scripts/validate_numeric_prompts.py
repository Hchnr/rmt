"""Compare eager/compiled cached logits on real benchmark prefill and decode."""
import json
from pathlib import Path
import torch
from rmt.inference.runner import Runner
from rmt.experts import project_qkv,project_output
from rmt.cache import RmtCapacityCache
r=Runner('artifacts/bootstrap/rmt-bound-4b',compiled=False,max_context=8192)
items=json.loads(Path('artifacts/v0.0.3/numeric_prompts.json').read_text())
bank=r.model.model.cell.bank;bank.compile_projections();fast=(bank._project_qkv,bank._project_output)
rows=[]
with torch.inference_mode():
 for start in range(0,len(items),4):
  batch=items[start:start+4];prompts=[r.render(x['messages']) for x in batch]
  enc=r.tokenizer(prompts,padding=True,return_tensors='pt').to(r.device)
  expected=[];continuation=[]
  for compiled in [False,True]:
   bank._project_qkv,bank._project_output=fast if compiled else (project_qkv,project_output)
   ids=enc.input_ids;mask=enc.attention_mask;cache=RmtCapacityCache(ids.shape[1]+8)
   for step in range(5):
    if compiled:torch.compiler.cudagraph_mark_step_begin()
    pos=(mask.cumsum(-1)-1).clamp_min(0)[:,-ids.shape[1]:]
    output=r.model(ids,attention_mask=mask,position_ids=pos,use_cache=True,past_key_values=cache,logits_to_keep=1).logits.float()
    if not compiled:
     expected.append(output.cpu());continuation.append(output[:,-1].argmax(-1)[:,None])
    else:
     reference=expected[step].to(r.device);delta=output-reference
     labels=continuation[step].reshape(-1)
     nll=torch.nn.functional.cross_entropy(output[:,0],labels)-torch.nn.functional.cross_entropy(reference[:,0],labels)
     row={'benchmark':batch[0]['benchmark'],'prompt_shapes':list(enc.input_ids.shape),'step':step,
       'max_abs':float(delta.abs().max()),'relative_rmse':float(delta.square().mean().sqrt()/reference.square().mean().sqrt()),'delta_nll':abs(float(nll)),
       'top1_equal':bool(torch.equal(output.argmax(-1),reference.argmax(-1)))}
     rows.append(row);print(row,flush=True)
    ids=continuation[step];mask=torch.cat((mask,torch.ones_like(ids)),1)
passed=all(x['relative_rmse']<=1e-3 and x['delta_nll']<=1e-3 for x in rows)
Path('reports/v0.0.3/real_prompt_numerics.json').write_text(json.dumps({'status':'passed' if passed else 'failed','prompt_count':len(items),'steps_per_batch':5,'rows':rows},indent=2)+'\n')
assert passed,'Real-prompt compiled numerics exceed fixed thresholds'

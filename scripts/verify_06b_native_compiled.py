"""Compare the actual native SDPA control against untrained compiled RMT."""
import json
from pathlib import Path
import torch
from rmt.inference.runner import Runner
from rmt.inference.cache import NativeCapacityCache
from rmt.cache import RmtCapacityCache

native=Runner('/share/project/eai_pwm/models/Qwen/Qwen3-0.6B',backend='qwen',compiled=False,attention='sdpa')
rmt=Runner('artifacts/v0.0.5/reference_hf',compiled=True,attention='sdpa')
rows=[]
with torch.inference_mode():
    for size in (1,4,8):
        prompts=[native.render([{'role':'user','content':' '.join(['Explain how to test sorting algorithms.']*(1+i*19))}]) for i in range(size)]
        batch=native.tokenizer(prompts,padding=True,return_tensors='pt').to('cuda')
        ids,mask=batch.input_ids,batch.attention_mask
        nc=NativeCapacityCache(ids.shape[1]+32);rc=RmtCapacityCache(ids.shape[1]+32)
        for step in range(32):
            positions=(mask.cumsum(-1)-1).clamp_min(0)[:,-ids.shape[1]:]
            kw=dict(attention_mask=mask,position_ids=positions,use_cache=True,logits_to_keep=1)
            x=native.model(ids,past_key_values=nc,**kw).logits.float()
            y=rmt.model(ids,past_key_values=rc,**kw).logits.float()
            delta=y-x
            rows.append({'batch_size':size,'step':step,'max_abs':delta.abs().max().item(),
                'relative_rmse':(delta.square().mean().sqrt()/x.square().mean().sqrt()).item(),
                'argmax_disagreements':(x.argmax(-1)!=y.argmax(-1)).sum().item()})
            ids=x[:,-1].argmax(-1,keepdim=True)
            mask=torch.cat((mask,torch.ones(size,1,device='cuda',dtype=mask.dtype)),1)
        del nc,rc
report={'status':'passed' if all(r['max_abs']<=.5 and r['relative_rmse']<.01 for r in rows) else 'failed',
    'all_bitwise_equal':all(r['max_abs']==0 for r in rows),'environment':native.engine_environment,
    'rows':rows,'scope':'Same explicit positions, heterogeneous left padding, native-greedy teacher-forced continuation, SDPA and capacity cache on both paths. Numerical limits retained from compiled decode verification; bitwise is measured separately.'}
Path('reports/v0.0.5/native_compiled_reference.json').write_text(json.dumps(report,indent=2)+'\n')
print({k:v for k,v in report.items() if k!='rows'},flush=True)
assert report['status']=='passed'

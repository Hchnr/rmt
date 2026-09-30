"""Check real 4B logits are unchanged and native EOS metadata controls stopping."""
import gc
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from rmt.inference.runner import Runner,Generation
from rmt.modeling_rmt import RmtForCausalLM
from rmt.evaluation.protocol import request_seed

torch.set_num_threads(4)
old='artifacts/v0.0.4_dynamic_recurr/train/teacher32_fixed36/hf'
new='artifacts/v0.0.4_dynamic_recurr/train/teacher32_fixed36_eos_fixed/hf'
r=Runner(new,compiled=False,attention='sdpa',max_context=40960)
reference=json.loads(Path('/share/project/eai_pwm/models/Qwen/Qwen3-4B/generation_config.json').read_text())
assert r.eos==set(reference['eos_token_id'])
before=RmtForCausalLM.from_pretrained(old,local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval()
ids=torch.tensor([[100,200,300,400]],device='cuda')
with torch.inference_mode():
 a=before(ids,use_cache=False,logits_to_keep=1).logits.clone()
 b=r.model(ids,use_cache=False,logits_to_keep=1).logits.clone()
assert torch.equal(a,b)
del before,a,b;gc.collect();torch.cuda.empty_cache()
r.model.model.cell.bank.compile_projections()
source=next(Path('artifacts/v0.0.4/datasets/full/ifeval').glob('*.jsonl'))
rows=list(map(json.loads,source.read_text().splitlines()))[28:32]
items=[]
for row in rows:
 messages=[{'role':'user','content':row['prompt']}]
 seed=request_seed([SimpleNamespace(role='user',content=row['prompt'])],17)
 items.append((row,r.render(messages),Generation(max_new_tokens=1024,temperature=.7,top_p=.8,top_k=20,presence_penalty=1.5,seed=seed)))
items.sort(key=lambda x:len(x[1]))
outputs=r.generate([x[1] for x in items],[x[2] for x in items])
summary=[]
for (row,prompt,cfg),out in zip(items,outputs):
 summary.append({'key':row['key'],'finish_reason':out['finish_reason'],'completion_tokens':out['completion_tokens'],
                 'last_token_id':out['token_ids'][-1],'token_ids_sha256':hashlib.sha256(json.dumps(out['token_ids']).encode()).hexdigest()})
result={'status':'passed','old_eos_token_ids':[151645],'corrected_eos_token_ids':sorted(r.eos),
        'forward_logits_bitwise_equal':True,'probe':summary,
        'observed_formerly_ignored_eos':sum(x['last_token_id']==151643 and x['finish_reason']=='stop' for x in summary),
        'scope':'Generation metadata only; one real 4B forward and four bounded real prompts. Probe output cap1024 differs from the frozen full benchmark32768, so this is a stopping diagnostic, not a task score.'}
Path('reports/v0.0.4_dynamic_recurr/generation_repair_verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

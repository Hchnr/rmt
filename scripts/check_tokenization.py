"""Verify both HF tokenizers encode every final evaluation prompt identically."""
import hashlib
import json
from pathlib import Path
from transformers import AutoTokenizer
r=AutoTokenizer.from_pretrained('artifacts/bootstrap/rmt-bound-4b',local_files_only=True)
q=AutoTokenizer.from_pretrained('/share/project/eai_pwm/models/Qwen/Qwen3-4B',local_files_only=True)
runs=json.loads(Path('reports/v0.0.3/representative_runs.json').read_text());rows=[]
for run in runs:
 if run['model']!='rmt':continue
 work=Path(run['work']);observed=set()
 for path in (work/'reviews').rglob('*.jsonl'):
  for line in path.read_text().splitlines():
   item=json.loads(line)
   assert item['input'].startswith('**User**: '), 'Expected the frozen zero-shot single-user protocol'
   messages=[{'role':'user','content':item['input'].removeprefix('**User**: ')}]
   rp=r.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
   qp=q.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
   assert rp==qp
   ri=r.encode(rp,add_special_tokens=False);qi=q.encode(qp,add_special_tokens=False);assert ri==qi
   sha=hashlib.sha256(rp.encode()).hexdigest();observed.add(sha)
   rows.append({'benchmark':run['benchmark'],'subset_file':path.name,'index':item['index'],'prompt_sha256':sha,
     'token_ids_sha256':hashlib.sha256(json.dumps(ri,separators=(',',':')).encode()).hexdigest(),'tokens':len(ri)})
 actual={json.loads(line)['rmt_metadata']['prompt_sha256'] for line in (work/'responses.jsonl').read_text().splitlines()}
 assert observed==actual,f"Reconstructed prompt mismatch: {run['benchmark']} observed={len(observed)} expected={len(actual)}"
Path('reports/v0.0.3/tokenization.json').write_text(json.dumps({'status':'passed','n':len(rows),'rows':rows},indent=2)+'\n');print('Matched',len(rows),'actual evaluation inputs')

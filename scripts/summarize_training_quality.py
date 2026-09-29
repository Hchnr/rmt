"""Compare frozen trained checkpoints against the original paired quick baseline."""
import hashlib
import json
from pathlib import Path
from rmt.evaluation.audit import audit
runs=[]
for candidate in ['fixed','opened']:
 runs+=json.loads(Path(f'reports/v0.0.4/quick_{candidate}_runs.json').read_text())
for row in json.loads(Path('reports/v0.0.3/representative_runs.json').read_text()):
 if row['model']=='qwen' and row['benchmark'] in ['math_500','ifeval']:runs.append({**row,'candidate':'original'})
results={};reviews={};identities={};protocols=[]
for row in runs:
 work=Path(row['work']);checked=audit(work,32);official=json.loads(next((work/'reports').rglob('*.json')).read_text())
 provenance=json.loads((work/'provenance.json').read_text());protocols.append(provenance['protocol']['generation'])
 results.setdefault(row['benchmark'],{})[row['candidate']]={'work':str(work),'official_score':official['score'],'audit':checked}
 by_prompt={}
 for path in (work/'reviews').rglob('*.jsonl'):
  for x in map(json.loads,path.open()):by_prompt[hashlib.sha256(x['input'].encode()).hexdigest()]=x['sample_score']['score']['value']
 reviews.setdefault(row['benchmark'],{})[row['candidate']]=by_prompt
 identities.setdefault(row['benchmark'],{})[row['candidate']]={(x['rmt_metadata']['prompt_sha256'],x['rmt_metadata']['seed']) for x in map(json.loads,(work/'responses.jsonl').open())}
assert all(p==protocols[0] for p in protocols)
paired={}
for benchmark,group in reviews.items():
 original=group['original'];assert len(original)==32
 assert all(x.keys()==original.keys() for x in group.values())
 assert all(x==identities[benchmark]['original'] for x in identities[benchmark].values())
 paired[benchmark]={name:sum(values[key]!=original[key] for key in original) for name,values in group.items()}
Path('reports/v0.0.4/trained_quality.json').write_text(json.dumps({'results':results,'changed_sample_scores_vs_original':paired,
 'same_prompts_and_request_seeds':True,'same_generation_protocol':True,
 'scope':'32 fixed samples per benchmark, non-thinking/2048 output; final checkpoints only, no benchmark-based selection; not a full benchmark claim'},indent=2)+'\n')
print({b:{c:r['official_score'] for c,r in group.items()} for b,group in results.items()})

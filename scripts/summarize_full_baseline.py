"""Audit full official benchmark runs and compare against paper non-thinking scores."""
import argparse
import hashlib
import json
from pathlib import Path
from rmt.evaluation.audit import audit
p=argparse.ArgumentParser();p.add_argument('--ifeval',required=True);p.add_argument('--math',required=True);a=p.parse_args()
results={}
for name,location,paper in [('ifeval',a.ifeval,81.2),('math_500',a.math,84.8)]:
 work=Path(location);provenance=json.loads((work/'provenance.json').read_text())
 checked=audit(work,provenance['selection']['count'])
 assert checked['unique_responses']==checked['review_count']
 official=json.loads(next((work/'reports').rglob('*.json')).read_text())
 # Compare shared sample identities, not arbitrary per-run sample indices.
 old=json.loads(Path('reports/v0.0.3/representative_runs.json').read_text())
 oldwork=Path(next(x['work'] for x in old if x['model']=='qwen' and x['benchmark']==name))
 def reviews(folder):
  output={}
  for path in (folder/'reviews').rglob('*.jsonl'):
   for row in map(json.loads,path.open()):output[hashlib.sha256(row['input'].encode()).hexdigest()]=row['sample_score']['score']['value']
  return output
 prev=reviews(oldwork);current=reviews(work);assert prev.keys()<=current.keys()
 results[name]={'work':str(work),'audit':checked,'official_report':official,'paper_score_percent':paper,
  'observed_score_percent':official['score']*100,'difference_percentage_points':official['score']*100-paper,
  'shared_previous_samples':len(prev),'shared_samples_score_changes':sum(prev[k]!=current[k] for k in prev),
  'shared_samples_before':prev,'shared_samples_now':{k:current[k] for k in prev},
  'metadata':provenance['metadata'],'generation':provenance['protocol']['generation']}
Path('reports/v0.0.4/full_baseline.json').write_text(json.dumps({'results':results,
 'paper':'https://arxiv.org/html/2505.09388v1#S4.T18',
 'scope':'Entire pinned IFEval and MATH-500 files, report sampling and output cap; prompt/version/RNG parity not guaranteed. HF SDPA replica topology is recorded; incomplete earlier eager attempts are excluded.'},indent=2)+'\n')
for name,r in results.items():print(name,r['observed_score_percent'],r['difference_percentage_points'],r['audit']['length_fraction'])

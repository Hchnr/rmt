"""Rescore old native answers and test ordering/thread invariance of RNG control."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import os
import random
os.environ['NLTK_DATA']=str(Path('artifacts/v0.0.3/nltk_data').resolve())
from rmt.evaluation.ifeval_rng import score_ifeval,install
root=Path('artifacts/v0.0.4/eval/full/qwen/ifeval/513cc7667b324bc8')
rows=[json.loads(x) for p in (root/'reviews'/'qwen').glob('*.jsonl') for x in p.read_text().splitlines()]
def measure(row):
 sample=row['sample_score'];return sample['sample_metadata']['key'],score_ifeval(sample['sample_metadata'],[sample['score']['extracted_prediction']],17)
state=random.getstate();first=dict(map(measure,rows));assert random.getstate()==state
random.seed(919);install(17)
with ThreadPoolExecutor(4) as pool:second=dict(pool.map(measure,reversed(rows)))
assert first==second
result={'status':'passed','source_work':str(root),'records':len(rows),'seed':17,'invariance':'sequential versus reverse order/4 threads; ambient Python RNG preserved',
 'metrics':{key:sum(x[key] for x in first.values())/len(first) for key in next(iter(first.values()))},
 'strict_correct':sum(x['prompt_level_strict'] for x in first.values()),
 'changed_from_historical':[{'id':r['sample_score']['sample_metadata']['key'],'old':r['sample_score']['score']['value'],'seeded':first[r['sample_score']['sample_metadata']['key']]} for r in rows if r['sample_score']['score']['value']!=first[r['sample_score']['sample_metadata']['key']]],
 'scores':first,'scope':'Original answers, official rules unchanged, per-record seeded Python fallback and language detection. Invalid punctuation fallback remains an upstream limitation; no custom checker correction.'}
Path('reports/v0.0.4_dynamic_recurr/ifeval_seeded_baseline.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ['scores','changed_from_historical']}),flush=True)

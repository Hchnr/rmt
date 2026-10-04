"""Diagnose unseeded official IFEval drift; score changes are recorded observations."""
import json
import os
from pathlib import Path
os.environ['NLTK_DATA']=str(Path('artifacts/v0.0.3/nltk_data').resolve())
from evalscope.benchmarks.ifeval.utils import process_results
root=Path('artifacts/v0.0.4/eval/full/qwen/ifeval/513cc7667b324bc8');rows=[]
for path in (root/'reviews'/'qwen').glob('*.jsonl'):
 for r in map(json.loads,path.read_text().splitlines()):
  score=r['sample_score'];old=score['score']['value'];new=process_results(score['sample_metadata'],[score['score']['extracted_prediction']])
  rows.append({'id':score['sample_metadata']['key'],'old':old,'new':new})
assert len(rows)==541 and len({x['id'] for x in rows})==541
mismatch=[x for x in rows if x['old']!=x['new']]
r={'status':'passed' if not mismatch else 'score_changed','source_work':str(root),'records':len(rows),'mismatches':mismatch,
 'old_strict_correct':sum(x['old']['prompt_level_strict'] for x in rows),'new_strict_correct':sum(x['new']['prompt_level_strict'] for x in rows),
 'scope':'All four official IFEval metrics rescored from saved original answers; no regeneration'}
Path('reports/v0.0.4_dynamic_recurr/ifeval_scorer_regression.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)

"""Rescore saved native baseline answers with the unchanged official EvalScope metric."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import signal
from evalscope.metrics.metric import Accuracy
root=Path('artifacts/v0.0.4/eval/full/qwen/math_500/4251241b6f9410b4');metric=Accuracy(numeric=True);rows=[]
def timeout(*args):raise TimeoutError('Scorer control exceeded 30 seconds; not counted as a wrong answer')
signal.signal(signal.SIGALRM,timeout)
for path in sorted((root/'reviews'/'qwen').glob('*.jsonl')):
 for row in map(json.loads,path.read_text().splitlines()):
  old=row['sample_score']['score'];signal.alarm(30)
  new=metric.apply([old['prediction']],[row['target']])[0];signal.alarm(0)
  rows.append({'id':row['sample_score']['sample_metadata']['question_id'],'old':old['value']['acc'],'new':new})
assert len(rows)==500 and len({x['id'] for x in rows})==500
mismatch=[x for x in rows if x['old']!=x['new']]
report={'status':'passed' if not mismatch else 'score_changed','source_work':str(root),'records':len(rows),
 'old_correct':sum(x['old'] for x in rows),'new_correct':sum(x['new'] for x in rows),'mismatches':mismatch,
 'versions':{x:importlib.metadata.version(x) for x in ['evalscope','latex2sympy2-extended','sympy','mpmath','antlr4-python3-runtime']},
 'scope':'Official Accuracy(numeric=True), saved original answers only; no regeneration, no score adapter replacement'}
Path('reports/v0.0.4_dynamic_recurr/math_scorer_regression.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
assert not mismatch

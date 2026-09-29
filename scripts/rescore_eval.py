"""Explicit review-only repair, preserving inference and original provenance."""
import argparse
import os
from pathlib import Path
import json
p=argparse.ArgumentParser();p.add_argument('work');a=p.parse_args();work=Path(a.work)
os.environ['NLTK_DATA']=str(Path('artifacts/v0.0.3/nltk_data').resolve())
from evalscope import TaskConfig,run_task
from rmt.evaluation.audit import audit
from rmt.evaluation import sandbox
sandbox.install()
from evalscope.api.benchmark import DefaultDataAdapter
original=DefaultDataAdapter.load_from_disk
DefaultDataAdapter.load_from_disk=lambda self,use_local_loader=False:original(self,use_local_loader=True)
config=TaskConfig.from_yaml(str(next((work/'configs').glob('*.yaml'))))
config.use_cache=str(work);config.rerun_review=True
result=run_task(config)
if sandbox.FAILURES:raise RuntimeError(sandbox.FAILURES)
provenance=json.loads((work/'provenance.json').read_text())
print(audit(work,provenance['selection']['count']))
(work/'rescore.json').write_text(json.dumps({'reason':'NLTK punkt resources installed; original inference reused','result':result},default=str,indent=2))

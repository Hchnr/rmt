"""Verify a completed run resumes after service restart with zero model calls."""
import argparse
import json
from pathlib import Path
import requests
from rmt.evaluation.run import file_hash,model_fingerprint
from rmt.evaluation.audit import audit
from evalscope import TaskConfig,run_task
from evalscope.models.openai_compatible import OpenAICompatibleAPI
from evalscope.api.benchmark import DefaultDataAdapter

p=argparse.ArgumentParser();p.add_argument('work');a=p.parse_args();work=Path(a.work)
provenance=json.loads((work/'provenance.json').read_text())
config=TaskConfig.from_yaml(str(next((work/'configs').glob('*.yaml'))))
session=requests.Session();session.trust_env=False
metadata=session.get(config.api_url.removesuffix('/v1')+'/metadata',timeout=10).json()
metadata['weights']=model_fingerprint(metadata['model'])
assert metadata==provenance['metadata'],'Restarted service differs from original provenance'
audit(work,provenance['selection']['count'])
paths=[*(work/'predictions').rglob('*.jsonl'),*(work/'reviews').rglob('*.jsonl'),work/'responses.jsonl']
before={str(p):file_hash(p) for p in paths}
def forbidden(*args,**kwargs):raise RuntimeError('Resume unexpectedly requested fresh inference')
OpenAICompatibleAPI.generate=forbidden
original=DefaultDataAdapter.load_from_disk
DefaultDataAdapter.load_from_disk=lambda self,use_local_loader=False:original(self,use_local_loader=True)
config.use_cache=str(work);config.rerun_review=False
run_task(config)
after={str(p):file_hash(p) for p in paths}
assert before==after,'Resume modified cached predictions/reviews'
result={'status':'passed','work':str(work),'model_calls':0,'immutable_output_files':len(paths),'service_metadata_match':True}
Path('reports/v0.0.3/resume.json').write_text(json.dumps(result,indent=2)+'\n');print(result)

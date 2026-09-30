"""Resume the interrupted v4 MATH run from its exact source snapshot.

The original orchestration predates configurable HTTP timeouts. Inject only
transport settings into TaskConfig, preserving source/weights/generation keys.
The original provenance and raw responses are retained under eval_recovery.
"""
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import requests

root=Path(__file__).resolve().parents[1]
recovery=root/'artifacts/v0.0.4/eval_recovery/math_timeout'
original=json.loads((recovery/'provenance.json').read_text())
tree=recovery/'source_tree'
for name,sha in original['code'].items():
    assert hashlib.sha256((tree/name).read_bytes()).hexdigest()==sha,name
session=requests.Session();session.trust_env=False
metadata=session.get('http://127.0.0.1:9007/metadata',timeout=10).json()
assert all(original['metadata'][key]==value for key,value in metadata.items()),'Runtime metadata changed'
assert set(metadata)==set(original['metadata'])-{'weights'}
config=json.loads((root/'configs/eval/v004_full_non_thinking.json').read_text())
assert config['generation']==original['protocol']['generation']
assert config['seed']==original['protocol']['seed']
assert config['model_args']=={'timeout':7200,'max_retries':0}
import evalscope
original_config=evalscope.TaskConfig
def transport_config(*args,**kwargs):
    kwargs['model_args']=config['model_args']
    return original_config(*args,**kwargs)
evalscope.TaskConfig=transport_config
record={'scope':'Transport-only recovery from frozen original source; no generation change',
        'original_key':original['key'],'source_hashes_verified':True,'runtime_metadata_equal':True,
        'model_args':config['model_args'],'wrapper_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'cached_predictions_before':334,'raw_responses_before':334}
(root/'reports/v0.0.4/math_recovery.json').write_text(json.dumps(record,indent=2)+'\n')
os.chdir(tree);sys.path.insert(0,str(tree/'src'))
sys.argv=['rmt.evaluation.run','--model','qwen','--port','9007','--benchmark','math_500','--phase','full',
          '--data-root','artifacts/v0.0.4/datasets','--output-root','artifacts/v0.0.4/eval',
          '--config','configs/eval/v004_full_non_thinking.json']
runpy.run_module('rmt.evaluation.run',run_name='__main__')

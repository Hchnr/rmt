"""Run frozen quick regressions on a completed training export, owning its service."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import requests
p=argparse.ArgumentParser();p.add_argument('--work',required=True);p.add_argument('--name',required=True)
p.add_argument('--gpu',required=True);p.add_argument('--port',type=int,default=9020)
p.add_argument('--report-root',default='reports/v0.0.4');p.add_argument('--output-root',default='artifacts/v0.0.4/quick_eval')
p.add_argument('--phase',choices=['pilot','representative','full'],default='representative')
p.add_argument('--checkpoint');p.add_argument('--attention',choices=['eager','sdpa']);p.add_argument('--config');a=p.parse_args()
a.attention=a.attention or ('sdpa' if a.phase=='full' else 'eager')
a.config=a.config or ('configs/eval/v004_full_non_thinking.json' if a.phase=='full' else 'configs/eval/quick_non_thinking.json')
prefix='full' if a.phase=='full' else 'quick'
max_context='40960' if a.phase=='full' else '8192'
Path(a.report_root).mkdir(parents=True,exist_ok=True)
work=Path(a.work);report=json.loads((work/'report.json').read_text());assert report['status']=='passed'
root=Path.cwd();server_env={**os.environ,'CUDA_VISIBLE_DEVICES':a.gpu,'PYTHONPATH':'src','TORCHINDUCTOR_COMPILE_THREADS':'2'}
log=Path(f'{a.report_root}/{prefix}_{a.name}_service.log').open('w')
server=subprocess.Popen([str(root/'.venv-cached/bin/python'),'-m','rmt.inference.server','--model',a.checkpoint or str(work/'hf'),'--attention',a.attention,'--name','rmt','--port',str(a.port),'--max-context',max_context],env=server_env,stdout=log,stderr=subprocess.STDOUT)
try:
 session=requests.Session();session.trust_env=False
 for _ in range(180):
  if server.poll() is not None:raise RuntimeError('Service exited')
  try:session.get(f'http://127.0.0.1:{a.port}/health',timeout=1).raise_for_status();break
  except requests.RequestException:time.sleep(1)
 else:raise TimeoutError('Service startup')
 results=[]
 for benchmark in ['math_500','ifeval']:
  path=Path(f'{a.report_root}/{prefix}_{a.name}_{benchmark}.log')
  env={**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONPATH':'src','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
  with path.open('w') as output:
   subprocess.run([str(root/'.venv-eval/bin/python'),'-m','rmt.evaluation.run','--model','rmt','--port',str(a.port),
    '--benchmark',benchmark,'--config',a.config,'--phase',a.phase,'--output-root',a.output_root],env=env,stdout=output,stderr=subprocess.STDOUT,check=True)
  location=next(x.split('=',1)[1] for x in reversed(path.read_text().splitlines()) if x.startswith('EVAL_WORK_DIR='))
  results.append({'candidate':a.name,'benchmark':benchmark,'work':location});print(results[-1],flush=True)
  Path(f'{a.report_root}/{prefix}_{a.name}_runs.json').write_text(json.dumps(results,indent=2)+'\n')
finally:
 server.terminate()
 try:server.wait(timeout=20)
 except subprocess.TimeoutExpired:server.kill();server.wait()
 log.close()

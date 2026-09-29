"""Run frozen quick regressions on a completed training export, owning its service."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import requests
p=argparse.ArgumentParser();p.add_argument('--work',required=True);p.add_argument('--name',required=True)
p.add_argument('--gpu',required=True);p.add_argument('--port',type=int,default=9020);a=p.parse_args()
work=Path(a.work);report=json.loads((work/'report.json').read_text());assert report['status']=='passed'
root=Path.cwd();server_env={**os.environ,'CUDA_VISIBLE_DEVICES':a.gpu,'PYTHONPATH':'src','TORCHINDUCTOR_COMPILE_THREADS':'2'}
log=Path(f'reports/v0.0.4/quick_{a.name}_service.log').open('w')
server=subprocess.Popen([str(root/'.venv-cached/bin/python'),'-m','rmt.inference.server','--model',str(work/'hf'),'--name','rmt','--port',str(a.port),'--max-context','8192'],env=server_env,stdout=log,stderr=subprocess.STDOUT)
try:
 session=requests.Session();session.trust_env=False
 for _ in range(180):
  if server.poll() is not None:raise RuntimeError('Service exited')
  try:session.get(f'http://127.0.0.1:{a.port}/health',timeout=1).raise_for_status();break
  except requests.RequestException:time.sleep(1)
 else:raise TimeoutError('Service startup')
 results=[]
 for benchmark in ['math_500','ifeval']:
  path=Path(f'reports/v0.0.4/quick_{a.name}_{benchmark}.log')
  env={**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONPATH':'src','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
  with path.open('w') as output:
   subprocess.run([str(root/'.venv-eval/bin/python'),'-m','rmt.evaluation.run','--model','rmt','--port',str(a.port),
    '--benchmark',benchmark,'--phase','representative','--output-root','artifacts/v0.0.4/quick_eval'],env=env,stdout=output,stderr=subprocess.STDOUT,check=True)
  location=next(x.split('=',1)[1] for x in reversed(path.read_text().splitlines()) if x.startswith('EVAL_WORK_DIR='))
  results.append({'candidate':a.name,'benchmark':benchmark,'work':location});print(results[-1],flush=True)
  Path(f'reports/v0.0.4/quick_{a.name}_runs.json').write_text(json.dumps(results,indent=2)+'\n')
finally:
 server.terminate()
 try:server.wait(timeout=20)
 except subprocess.TimeoutExpired:server.kill();server.wait()
 log.close()

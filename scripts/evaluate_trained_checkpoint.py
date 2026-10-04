"""Run frozen quick regressions on a completed training export, owning its service."""
import argparse
import json
import hashlib
import os
from pathlib import Path
import subprocess
import signal
import time
import requests
p=argparse.ArgumentParser();p.add_argument('--work',required=True);p.add_argument('--name',required=True)
p.add_argument('--gpu',required=True);p.add_argument('--port',type=int,default=9020)
p.add_argument('--report-root',default='reports/v0.0.4');p.add_argument('--output-root',default='artifacts/v0.0.4/quick_eval')
p.add_argument('--phase',choices=['pilot','representative','full'],default='representative')
p.add_argument('--benchmarks',nargs='+',choices=['math_500','ifeval'],default=['math_500','ifeval'])
p.add_argument('--response-timeout',type=float);p.add_argument('--recover-work');p.add_argument('--eval-batch-size',type=int);p.add_argument('--data-root');p.add_argument('--checkpoint');p.add_argument('--attention',choices=['eager','sdpa']);p.add_argument('--config');a=p.parse_args()
if a.recover_work and len(a.benchmarks)!=1:raise ValueError('Recovery must name exactly one benchmark')
a.attention=a.attention or ('sdpa' if a.phase=='full' else 'eager')
a.config=a.config or ('configs/eval/v004_dynamic_full_non_thinking.json' if a.phase=='full' else 'configs/eval/quick_non_thinking.json')
a.data_root=a.data_root or ('artifacts/v0.0.4/datasets' if a.phase=='full' else 'artifacts/v0.0.3/datasets')
prefix='full' if a.phase=='full' else 'quick'
max_context='40960' if a.phase=='full' else '8192'
Path(a.report_root).mkdir(parents=True,exist_ok=True)
work=Path(a.work);report=json.loads((work/'report.json').read_text());assert report['status']=='passed'
root=Path.cwd();gpus=a.gpu.split(',')
if len(set(gpus))!=len(gpus) or not set(gpus)<=set(map(str,range(8))):
 raise ValueError('GPU IDs must be unique authorized IDs 0-7')
a.eval_batch_size=4*len(gpus) if a.eval_batch_size is None else a.eval_batch_size
if a.eval_batch_size<1:raise ValueError('Positive evaluation concurrency required')
processes=[];logs=[];service_started=time.monotonic();completed=False
signal.signal(signal.SIGTERM,lambda *args: (_ for _ in ()).throw(KeyboardInterrupt()))
def launch(command,env,suffix):
 log=Path(f'{a.report_root}/{prefix}_{a.name}_{suffix}.log').open('w');logs.append(log)
 process=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT);processes.append(process)
 return process
session=requests.Session();session.trust_env=False
def ready(port,process):
 for _ in range(180):
  if process.poll() is not None:raise RuntimeError(f'Service {port} exited')
  try:session.get(f'http://127.0.0.1:{port}/health',timeout=1).raise_for_status();return
  except requests.RequestException:time.sleep(1)
 raise TimeoutError(f'Service {port} startup')
transport={'response_wait_seconds':a.response_timeout if a.response_timeout is not None else 3600,'wrapper_sha256':hashlib.sha256(Path('scripts/serve_eval_transport.py').read_bytes()).hexdigest() if a.response_timeout is not None else None,'scope':'Transport waiting only; no model, logits, sampler, generation cap or seed changes.'}
Path(f'{a.report_root}/{prefix}_{a.name}_transport.json').write_text(json.dumps(transport,indent=2)+'\n')
try:
 ports=[a.port] if len(gpus)==1 else list(range(a.port+1,a.port+1+len(gpus)))
 for gpu,port in zip(gpus,ports):
  env={**os.environ,'CUDA_VISIBLE_DEVICES':gpu,'PYTHONPATH':'src','TORCHINDUCTOR_COMPILE_THREADS':'2'}
  entry=([str(root/'.venv-cached/bin/python'),'scripts/serve_eval_transport.py','--response-timeout',str(a.response_timeout)] if a.response_timeout is not None else [str(root/'.venv-cached/bin/python'),'-m','rmt.inference.server'])
  launch(entry+['--model',a.checkpoint or str(work/'hf'),
   '--attention',a.attention,'--name','rmt','--port',str(port),'--max-context',max_context],env,f'service_{port}')
 for port,process in zip(ports,processes):ready(port,process)
 if len(gpus)>1:
  pool=launch([str(root/'.venv-cached/bin/python'),'scripts/serve_eval_pool.py',
   '--ports',','.join(map(str,ports)),'--port',str(a.port)],{**os.environ,'CUDA_VISIBLE_DEVICES':''},'pool')
  ready(a.port,pool)
 results=[]
 for benchmark in a.benchmarks:
  path=Path(f'{a.report_root}/{prefix}_{a.name}_{benchmark}.log')
  env={**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONPATH':'src','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
  with path.open('w') as output:
   entry=([str(root/'.venv-eval/bin/python'),'scripts/recover_eval_responses.py','--work',a.recover_work,'--report',f'{a.report_root}/{prefix}_{a.name}_recovery.json','--'] if a.recover_work else [str(root/'.venv-eval/bin/python'),'-m','rmt.evaluation.run'])
   subprocess.run(entry+['--model','rmt','--port',str(a.port),
    '--benchmark',benchmark,'--config',a.config,'--phase',a.phase,'--data-root',a.data_root,'--eval-batch-size',str(a.eval_batch_size),'--output-root',a.output_root],env=env,stdout=output,stderr=subprocess.STDOUT,check=True)
  location=next(x.split('=',1)[1] for x in reversed(path.read_text().splitlines()) if x.startswith('EVAL_WORK_DIR='))
  results.append({'candidate':a.name,'benchmark':benchmark,'work':location});print(results[-1],flush=True)
  Path(f'{a.report_root}/{prefix}_{a.name}_runs.json').write_text(json.dumps(results,indent=2)+'\n')
 completed=True
finally:
 for process in reversed(processes):
  if process.poll() is None:process.terminate()
 for process in reversed(processes):
  try:process.wait(timeout=20)
  except subprocess.TimeoutExpired:process.kill();process.wait()
 for log in logs:log.close()
 elapsed=time.monotonic()-service_started
 Path(f'{a.report_root}/{prefix}_{a.name}_cost.json').write_text(json.dumps({'status':'completed' if completed else 'failed','gpus':gpus,'wall_seconds':elapsed,'reserved_gpu_hours':elapsed*len(gpus)/3600,'scope':'Allocated replica lifetime including loading, scoring waits and cleanup; not GPU kernel utilization'},indent=2)+'\n')

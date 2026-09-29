"""Run explicitly allocated single-GPU model replicas; Ctrl-C stops owned children."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

p=argparse.ArgumentParser();p.add_argument('--gpus',default='0,1,2,3,4,5,6,7');p.add_argument('--port-base',type=int,default=8901)
p.add_argument('--rmt',default='artifacts/bootstrap/rmt-bound-4b');p.add_argument('--qwen',default='/share/project/eai_pwm/models/Qwen/Qwen3-4B')
a=p.parse_args();gpus=a.gpus.split(',')
if not set(gpus)<=set(map(str,range(8))) or len(set(gpus))!=len(gpus):raise ValueError('Use unique authorized physical GPU IDs 0–7')
processes=[];handles=[];inventory=[]
try:
 for i,gpu in enumerate(gpus):
  model='rmt' if i%2==0 else 'qwen';port=a.port_base+i
  path=a.rmt if model=='rmt' else a.qwen
  log=Path(f'reports/v0.0.3/replica_{port}.log');log.parent.mkdir(parents=True,exist_ok=True)
  handle=log.open('w');handles.append(handle)
  env={**os.environ,'CUDA_VISIBLE_DEVICES':gpu,'TORCHINDUCTOR_COMPILE_THREADS':'4'}
  command=['.venv-cached/bin/python','-m','rmt.inference.server','--model',path,'--name',model,'--port',str(port),'--max-context','8192']
  if model=='qwen':command+=['--backend','qwen','--eager']
  proc=subprocess.Popen(command,env=env,stdout=handle,stderr=subprocess.STDOUT);processes.append(proc)
  inventory.append({'gpu':gpu,'model':model,'port':port,'pid':proc.pid})
 Path('artifacts/v0.0.3').mkdir(parents=True,exist_ok=True)
 Path('artifacts/v0.0.3/replicas.json').write_text(json.dumps(inventory,indent=2))
 print(json.dumps(inventory),flush=True)
 signal.signal(signal.SIGTERM,lambda *args: (_ for _ in ()).throw(KeyboardInterrupt()))
 while all(p.poll() is None for p in processes):time.sleep(1)
 raise RuntimeError('A replica exited; see its log')
except KeyboardInterrupt:pass
finally:
 for proc in processes:
  if proc.poll() is None:proc.terminate()
 for proc in processes:
  try:proc.wait(timeout=15)
  except subprocess.TimeoutExpired:proc.kill();proc.wait()
 for handle in handles:handle.close()

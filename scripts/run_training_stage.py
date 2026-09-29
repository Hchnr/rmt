"""Run explicit training configurations sequentially on authorized GPUs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import yaml
p=argparse.ArgumentParser();p.add_argument('--gpus',required=True);p.add_argument('--configs',nargs='+',required=True)
p.add_argument('--verify-restart',action='store_true');a=p.parse_args()
env={**os.environ,'CUDA_VISIBLE_DEVICES':a.gpus,'PYTHONPATH':'src','OMP_NUM_THREADS':'2','TORCHINDUCTOR_COMPILE_THREADS':'2'}
results=[]
for config in a.configs:
 cfg=yaml.safe_load(Path(config).read_text());name=Path(config).stem
 cmd=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node='+str(len(a.gpus.split(','))),'-m','rmt.train','--config',config]
 start=time.monotonic()
 with Path(f'reports/v0.0.4/{name}.log').open('w') as log:subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 if a.verify_restart and cfg.get('verify_replay',True):
  with Path(f'reports/v0.0.4/{name}_fresh_resume.log').open('w') as log:subprocess.run(cmd+['--verify-resume'],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 results.append({'config':config,'report':cfg['report'],'wall_seconds':time.monotonic()-start})
 Path('reports/v0.0.4/training_stage_runs.json').write_text(json.dumps(results,indent=2)+'\n')
 print(json.dumps(results[-1]),flush=True)

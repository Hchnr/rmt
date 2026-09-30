"""Run explicit training configurations sequentially on authorized GPUs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone
import yaml
p=argparse.ArgumentParser();p.add_argument('--gpus',required=True);p.add_argument('--configs',nargs='+',required=True)
p.add_argument('--verify-restart',action='store_true');p.add_argument('--report-root',default='reports/v0.0.4');a=p.parse_args()
report_root=Path(a.report_root);report_root.mkdir(parents=True,exist_ok=True)
env={**os.environ,'CUDA_VISIBLE_DEVICES':a.gpus,'PYTHONPATH':'src','OMP_NUM_THREADS':'2','TORCHINDUCTOR_COMPILE_THREADS':'2'}
results=[]
archive=(report_root/'training_stages')/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')+'.json')
archive.parent.mkdir(parents=True,exist_ok=True)
def persist():
 text=json.dumps(results,indent=2)+'\n'
 archive.write_text(text)
 (report_root/'training_stage_runs.json').write_text(text)
for config in a.configs:
 cfg=yaml.safe_load(Path(config).read_text());name=Path(config).stem
 cmd=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node='+str(len(a.gpus.split(','))),'-m','rmt.train','--config',config]
 start=time.monotonic()
 row={'config':config,'report':cfg['report'],'gpus':a.gpus,'status':'running','archive':str(archive)}
 results.append(row);persist()
 try:
  with (report_root/f'{name}.log').open('w') as log:subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
  if a.verify_restart and cfg.get('verify_replay',True):
   with (report_root/f'{name}_fresh_resume.log').open('w') as log:subprocess.run(cmd+['--verify-resume'],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 except subprocess.CalledProcessError as error:
  row.update(status='failed',returncode=error.returncode,wall_seconds=time.monotonic()-start);persist();raise
 row.update(status='passed',wall_seconds=time.monotonic()-start);persist()
 print(json.dumps(row),flush=True)

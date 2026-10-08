"""Run a bounded stage on physical GPUs 4–7 with durable reservation accounting."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--name',required=True)
    p.add_argument('--gpus',required=True);p.add_argument('--timeout',type=float,default=3600)
    p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
    devices=a.gpus.split(',')
    if not devices or len(devices)!=len(set(devices)) or not set(devices)<=set('4567'):
        raise ValueError('Only distinct physical GPUs 4–7 are authorized')
    command=a.command[1:] if a.command[:1]==['--'] else a.command
    if not command:raise ValueError('Missing command')
    root=Path('reports/v0.0.5/stages');root.mkdir(parents=True,exist_ok=True)
    target=root/(a.name+'.json')
    if target.exists():raise ValueError('Stage name already exists; preserve previous accounting')
    now=lambda:datetime.now(timezone.utc).isoformat()
    record=dict(name=a.name,gpus=devices,command=command,started_at=now(),status='running',
                commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    target.write_text(json.dumps(record,indent=2)+'\n')
    env={**os.environ,'RMT_ALLOWED_GPUS':'4,5,6,7','CUDA_VISIBLE_DEVICES':a.gpus,
         'TORCHINDUCTOR_COMPILE_THREADS':'2'}
    started=time.monotonic()
    with Path(f'reports/v0.0.5/{a.name}.log').open('w') as log:
        process=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            code=process.wait(timeout=a.timeout)
            record.update(status='passed' if code==0 else 'failed',returncode=code)
        except BaseException as error:
            import signal
            os.killpg(process.pid,signal.SIGTERM)
            try:process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL);process.wait()
            record.update(status='interrupted',error=type(error).__name__)
            raise
        finally:
            record.update(ended_at=now(),wall_seconds=time.monotonic()-started)
            record['reserved_gpu_hours']=record['wall_seconds']*len(devices)/3600
            target.write_text(json.dumps(record,indent=2)+'\n')
    if code:raise SystemExit(code)
    print(json.dumps(record),flush=True)


if __name__=='__main__':main()

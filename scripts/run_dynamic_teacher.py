"""Own a bounded set of teacher replica processes and summarize actual generation cost."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
p=argparse.ArgumentParser();p.add_argument('--gpus',required=True);p.add_argument('--limit',type=int,default=1000)
p.add_argument('--batch-size',type=int,default=4);p.add_argument('--max-new-tokens',type=int,default=1024)
p.add_argument('--output',default='artifacts/v0.0.4_dynamic_recurr/teacher_answers');a=p.parse_args()
gpus=a.gpus.split(',');root=Path(a.output);root.mkdir(parents=True,exist_ok=True)
processes=[];logs=[];start=time.monotonic()
try:
 for rank,gpu in enumerate(gpus):
  log=(root/f'rank_{rank}.log').open('w');logs.append(log)
  cmd=[sys.executable,'scripts/generate_dynamic_teacher.py','--rank',str(rank),'--world-size',str(len(gpus)),
       '--limit',str(a.limit),'--batch-size',str(a.batch_size),'--max-new-tokens',str(a.max_new_tokens),'--output',a.output]
  processes.append(subprocess.Popen(cmd,env={**os.environ,'CUDA_VISIBLE_DEVICES':gpu,'OMP_NUM_THREADS':'4'},stdout=log,stderr=subprocess.STDOUT))
 while any(p.poll() is None for p in processes):
  failed=[i for i,p in enumerate(processes) if p.poll() not in (None,0)]
  if failed:raise RuntimeError('Teacher worker failed: '+str(failed))
  time.sleep(2)
 if any(p.returncode for p in processes):raise RuntimeError('Teacher worker failed')
 summaries=[json.loads((root/f'rank_{i}_summary.json').read_text()) for i in range(len(gpus))]
 report={'status':'completed','gpus':gpus,'wall_seconds':time.monotonic()-start,'workers':summaries,
    'worker_gpu_hours':sum(x['wall_seconds'] for x in summaries)/3600,
    'output_tokens':sum(x['output_tokens'] for x in summaries),'truncated_records':sum(x['truncated_records'] for x in summaries)}
 (root/'generation_summary.json').write_text(json.dumps(report,indent=2)+'\n')
 Path('reports/v0.0.4_dynamic_recurr/teacher_generation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
finally:
 for process in processes:
  if process.poll() is None:process.terminate()
 for process in processes:
  try:process.wait(timeout=20)
  except subprocess.TimeoutExpired:process.kill();process.wait()
 for log in logs:log.close()

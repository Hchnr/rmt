"""Validate real 0.6B FSDP on GPU4/5 after the native evaluation releases them."""
import json
from pathlib import Path
import subprocess
import time
started=time.monotonic()
while True:
    row=json.loads(Path('reports/v0.0.5/stages/native_full_v2.json').read_text())
    if row.get('ended_at'):
        if row['status']!='passed':raise RuntimeError('Native evaluation failed')
        break
    if time.monotonic()-started>7200:raise TimeoutError('Native evaluation release')
    time.sleep(15)
for name,extra in [('fsdp06b_smoke',[]),('fsdp06b_fresh_resume',['--verify-resume'])]:
    subprocess.run(['.venv-cached/bin/python','scripts/run_06b_stage.py','--name',name,'--gpus','4,5','--timeout','1800','--',
        '.venv-cached/bin/torchrun','--standalone','--nproc-per-node=2','-m','rmt.train',
        '--config','configs/v0.0.5/fsdp06b_smoke.yaml',*extra],check=True)

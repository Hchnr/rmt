"""Reproduce frozen recipes with one deterministic512-token development protocol."""
import argparse
import json
from pathlib import Path
import subprocess
import time

p=argparse.ArgumentParser();p.add_argument('--gpu',choices=['6','7'],required=True);a=p.parse_args()
prerequisite='short_hf_reload' if a.gpu=='6' else 'native_compiled_reference_aligned'
names=(['pilot_ce','pilot_kl01','pilot_kl05','short_kl05'] if a.gpu=='6'
       else ['native','untrained','mixed_kl05','mixed_kl05_lr1e6'])
started=time.monotonic()
while True:
    try:record=json.loads((Path('reports/v0.0.5/stages')/(prerequisite+'.json')).read_text())
    except (FileNotFoundError,json.JSONDecodeError):record={}
    if record.get('ended_at'):
        if record['status']!='passed':raise RuntimeError(f'Prerequisite {prerequisite} failed')
        break
    if time.monotonic()-started>21600:raise TimeoutError('Waiting for GPU release/checks')
    time.sleep(15)
for name in names:
    model=('/share/project/eai_pwm/models/Qwen/Qwen3-0.6B' if name=='native'
           else 'artifacts/v0.0.5/reference_hf' if name=='untrained' else f'artifacts/v0.0.5/{name}/hf')
    command=['.venv-cached/bin/python','scripts/evaluate_06b_development.py','--model',model,
        '--backend','qwen' if name=='native' else 'rmt','--max-new-tokens','512','--deterministic',
        '--output',f'reports/v0.0.5/{name}_development_deterministic512.json']
    if name!='native':command.append('--compile')
    subprocess.run(['.venv-cached/bin/python','scripts/run_06b_stage.py','--name','detdev_'+name,
        '--gpus',a.gpu,'--timeout','1200','--',*command],check=True)
print('Frozen development recipes rechecked; no weights or data changed.',flush=True)

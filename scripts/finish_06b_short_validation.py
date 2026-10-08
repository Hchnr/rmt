"""Run pending independent checks only after the owning evaluation releases GPUs."""
import json
from pathlib import Path
import subprocess
import time

stage=Path('reports/v0.0.5/stages/candidate_full.json')
while True:
    try:
        status=json.loads(stage.read_text())
    except (FileNotFoundError,json.JSONDecodeError):
        time.sleep(15)
        continue
    if status.get('ended_at') and status['status']!='running':
        break
    time.sleep(15)
commands=[('short_fresh_resume',['.venv-cached/bin/python','-m','rmt.train','--config','configs/v0.0.5/short_kl05.yaml','--verify-resume']),
          ('short_hf_reload',['.venv-cached/bin/python','scripts/validate_training_export.py','artifacts/v0.0.5/short_kl05'])]
for name,command in commands:
    subprocess.run(['.venv-cached/bin/python','scripts/run_06b_stage.py','--name',name,'--gpus','6','--timeout','1200','--',*command],check=True)
root=Path('reports/v0.0.5')
report=json.loads((root/'training_export_resume.json').read_text())
for row in report['rows']:
    if row['name']=='short_kl05':
        for key in ['fresh_resume','hf_reload']:
            row[key]=json.loads((Path('artifacts/v0.0.5/short_kl05')/(key+'.json')).read_text())
            assert row[key]['status']=='passed'
(root/'training_export_resume.json').write_text(json.dumps(report,indent=2)+'\n')
print('Pending short-run checks passed; no training exposure added.',flush=True)

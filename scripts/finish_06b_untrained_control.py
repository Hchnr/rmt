"""Run the frozen untrained control after GPU6/7 validation releases devices."""
import json
from pathlib import Path
import subprocess
import time

started=time.monotonic()
for name in ['detdev_short_kl05','deterministic_dynamic_reference']:
    while True:
        path=Path('reports/v0.0.5/stages')/(name+'.json')
        record=json.loads(path.read_text()) if path.exists() else {}
        if record.get('ended_at'):
            if record['status']!='passed':raise RuntimeError(f'{name} failed')
            break
        if time.monotonic()-started>7200:raise TimeoutError('Waiting for validation release')
        time.sleep(15)
work=Path('artifacts/v0.0.5/untrained_reference');work.mkdir(exist_ok=True)
(work/'report.json').write_text(json.dumps({'status':'passed','scope':'Untrained reference; status denotes P0 validation, not training.',
    'reference_report':'reports/v0.0.5/reference.json','aligned_backend_report':'reports/v0.0.5/native_compiled_reference_aligned.json'},indent=2)+'\n')
subprocess.run(['.venv-cached/bin/python','scripts/run_06b_stage.py','--name','untrained_full','--gpus','6,7','--timeout','14400','--',
    '.venv-cached/bin/python','scripts/evaluate_trained_checkpoint.py','--work',str(work),'--name','untrained06b','--gpu','6,7','--port','9230',
    '--backend','rmt','--checkpoint','artifacts/v0.0.5/reference_hf','--phase','full','--config','configs/v0.0.5/full_non_thinking.json',
    '--report-root','reports/v0.0.5','--output-root','artifacts/v0.0.5/eval','--response-timeout','7200'],check=True)

"""Finalize paired reports only after all frozen full-evaluation stages pass."""
import json
import os
from pathlib import Path
import subprocess
import time

root=Path('reports/v0.0.5')
started=time.monotonic()
for name in ['native_full_v2','candidate_full','untrained_full']:
    while True:
        path=root/'stages'/(name+'.json')
        try:row=json.loads(path.read_text())
        except (FileNotFoundError,json.JSONDecodeError):row={}
        if row.get('ended_at'):
            if row['status']!='passed':raise RuntimeError(f'{name} did not pass')
            break
        if time.monotonic()-started>18000:raise TimeoutError('Full evaluation completion')
        time.sleep(15)
base=root/'full_native06b_v2_runs.json'
untrained=root/'full_untrained06b_runs.json'
trained=root/'full_mixed_lr1e6_final_runs.json'
for reference,candidate,output,label in [(base,trained,'formal_comparison','native'),
    (base,untrained,'inference_effect_comparison','native'),
    (untrained,trained,'training_effect_comparison','untrained_rmt')]:
    subprocess.run(['.venv-eval/bin/python','scripts/summarize_06b_formal.py','--reference',str(reference),
        '--candidate',str(candidate),'--output',str(root/(output+'.json')),'--reference-label',label],
        env={**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONPATH':'src'},check=True)
subprocess.run(['.venv-cached/bin/python','scripts/summarize_06b_campaign.py'],check=True)
print('All three formal paired comparisons audited; no generation or training was changed.',flush=True)

"""Finalize paired reports only after all frozen full-evaluation stages pass."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

parser=argparse.ArgumentParser()
parser.add_argument('--control-stages',nargs='+',default=['untrained_full'])
parser.add_argument('--control-manifests',nargs='+',default=['reports/v0.0.5/full_untrained06b_runs.json'])
args=parser.parse_args()
root=Path('reports/v0.0.5')
started=time.monotonic()
for name in ['native_full_v2','candidate_full',*args.control_stages]:
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
controls=[row for path in args.control_manifests for row in json.loads(Path(path).read_text())]
assert len(controls)==2 and {row['benchmark'] for row in controls}=={'math_500','ifeval'}
untrained.write_text(json.dumps(controls,indent=2)+'\n')
for reference,candidate,output,label in [(base,trained,'formal_comparison','native'),
    (base,untrained,'inference_effect_comparison','native'),
    (untrained,trained,'training_effect_comparison','untrained_rmt')]:
    subprocess.run(['.venv-eval/bin/python','scripts/summarize_06b_formal.py','--reference',str(reference),
        '--candidate',str(candidate),'--output',str(root/(output+'.json')),'--reference-label',label],
        env={**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONPATH':'src'},check=True)
subprocess.run(['.venv-cached/bin/python','scripts/summarize_06b_campaign.py'],check=True)
print('All three formal paired comparisons audited; no generation or training was changed.',flush=True)

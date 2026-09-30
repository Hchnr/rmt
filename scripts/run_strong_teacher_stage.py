"""Filter the completed frozen teacher corpus, then own two paired training queues."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

p = argparse.ArgumentParser()
p.add_argument('--wait-seconds', type=int, default=0)
a = p.parse_args()
root = Path.cwd()
reports = root/'reports/v0.0.4_dynamic_recurr'
summary = root/'artifacts/v0.0.4_dynamic_recurr/teacher_answers/generation_summary.json'
deadline = time.monotonic() + a.wait_seconds
while not summary.exists():
    if time.monotonic() >= deadline:
        raise TimeoutError('Teacher has not completed; no training started')
    time.sleep(5)
assert json.loads(summary.read_text())['status'] == 'completed'
# The teacher owner writes this only after all worker processes have exited.
result_path = reports/'strong_teacher_stage.json'
if result_path.exists():
    raise FileExistsError('Preserve the existing stage result; resume individual configs explicitly')
env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '',
       'PYTHONPATH': 'artifacts/v0.0.4_dynamic_recurr/math_verifier:src'}
with (reports/'teacher_filter.log').open('w') as log:
    subprocess.run([str(root/'.venv-eval/bin/python'), 'scripts/prepare_teacher_distillation.py'],
                   env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
filtered = json.loads((reports/'teacher_distillation.json').read_text())
if filtered['accepted'] < 64:
    raise ValueError('Insufficient accepted teacher records for the frozen experiment')
queues = [('0,1,2,3', ['teacher32_fixed36', 'teacher32_fixed40', 'teacher32_anchor40']),
          ('4,5,6,7', ['teacher32_hybrid', 'matched_source_fixed36'])]
children = []; logs = []; started = time.monotonic()
result = {'status': 'running', 'accepted': filtered['accepted'], 'queues': queues}
def persist():
    result_path.write_text(json.dumps(result, indent=2)+'\n')
persist()
try:
    for i, (gpus, names) in enumerate(queues):
        log = (reports/f'teacher_training_queue_{i}.log').open('w'); logs.append(log)
        command = [str(root/'.venv-cached/bin/python'), 'scripts/run_training_stage.py',
                   '--gpus', gpus, '--verify-restart', '--report-root', str(reports),
                   '--configs', *[f'configs/dynamic/{name}.yaml' for name in names]]
        children.append(subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True))
    while any(child.poll() is None for child in children):
        if any(child.poll() not in (None, 0) for child in children):
            raise RuntimeError('A training queue failed; inspect its immutable stage archive and logs')
        time.sleep(5)
    if any(child.returncode for child in children):
        raise RuntimeError('A training queue failed')
    result['status'] = 'passed'
except BaseException:
    result['status'] = 'failed'
    raise
finally:
    for child in children:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
    for child in children:
        try: child.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL); child.wait()
    for log in logs: log.close()
    result['wall_seconds'] = time.monotonic()-started
    persist()
print(json.dumps(result), flush=True)

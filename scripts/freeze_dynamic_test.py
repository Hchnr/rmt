"""Freeze the predeclared final test candidates before any test scoring."""
import hashlib,json
from pathlib import Path
from datetime import datetime,timezone
from rmt.evaluation.run import model_fingerprint
target=Path('reports/v0.0.4_dynamic_recurr/frozen_test_protocol.json')
if target.exists():raise FileExistsError('Do not overwrite a frozen test protocol')
root=Path('artifacts/v0.0.4_dynamic_recurr/train')
names={'native':('/share/project/eai_pwm/models/Qwen/Qwen3-4B',True),'initial_fixed36':(str(root/'fixed36/hf'),False),'initial_hybrid':(str(root/'hybrid/hf'),False)}
for name in ['teacher32_fixed36','teacher32_fixed40','teacher32_anchor40','matched_source_fixed36']:
 names[name]=(str(root/name/'hf'),False)
names['teacher32_hybrid']=(str(root/'teacher32_hybrid_calibrated/hf'),False)
rows=[{'name':name,'model':path,'original':original,'fingerprint':model_fingerprint(path)} for name,(path,original) in names.items()]
report={'frozen_at_utc':datetime.now(timezone.utc).isoformat(),'status':'frozen_before_test_scoring','models':rows,
 'test_data':'artifacts/v0.0.4_dynamic_recurr/corpus/test.jsonl',
 'test_sha256':hashlib.sha256(Path('artifacts/v0.0.4_dynamic_recurr/corpus/test.jsonl').read_bytes()).hexdigest(),
 'core_code':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('src/rmt').glob('*.py')},
 'primary_full_benchmarks':['teacher32_fixed36','teacher32_hybrid'],
 'selection':'Predeclared strong-teacher fixed36 versus H+P pair; other models are held-out CE controls. Not a claim that either is the best dev checkpoint.',
 'halting':'Keep original H=.2/P=.05, full calibration mean40.0104; frozen variant changes no config values or weights.',
 'prohibition':'Do not tune weights, thresholds, or candidate selection from test or final benchmark scores.'}
Path('reports/v0.0.4_dynamic_recurr/frozen_test_protocol.json').write_text(json.dumps(report,indent=2)+'\n')

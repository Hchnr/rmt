"""Separate observed stage reservations from conservative whole-interval ceilings."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

root=Path('reports/v0.0.4_dynamic_recurr')
training=[]
for path in sorted((root/'training_stages').glob('*.json')):
    for row in json.loads(path.read_text()):
        if row['status']!='passed':raise ValueError('Incomplete or failed training stage in final cost audit')
        training.append({'config':row['config'],'archive':str(path),
                         'gpu_hours':row['wall_seconds']*len(row['gpus'].split(','))/3600})
evaluations=[]
for path in sorted(root.glob('full_teacher32*_cost.json')):
    row=json.loads(path.read_text())
    if row['status']!='completed':raise ValueError('Failed evaluation requires separate attempt accounting')
    evaluations.append({'report':str(path),'gpu_hours':row['reserved_gpu_hours']})
if len(evaluations)!=6:raise ValueError('All six preregistered full tasks must complete first')
failed_attempts=[{'report':str(p),'gpu_hours':json.loads(p.read_text())['cost']['reserved_gpu_hours']} for p in sorted((root/'eval_attempts').glob('*.json'))]
pilots=[]
for name in ['teacher_pilot','teacher_batch8_probe']:
    path=Path('artifacts/v0.0.4_dynamic_recurr')/name/'rank_0_summary.json'
    row=json.loads(path.read_text());pilots.append({'name':name,'gpu_hours':row['wall_seconds']/3600})
teacher=json.loads((root/'teacher_cost.json').read_text())
# This repository baseline predates task work, so charging all eight cards for
# its whole interval is conservative, including idle time and CPU/download work.
start_text=subprocess.check_output(['git','show','-s','--format=%cI','2ff4f41'],text=True).strip()
start=datetime.fromisoformat(start_text);end=datetime.now(timezone.utc)
ceiling=(end-start).total_seconds()*8/3600
result={'as_of_utc':end.isoformat(),'known_training_reservations':training,
        'known_full_eval_reservations':evaluations,'failed_eval_attempts':failed_attempts,'teacher_pilots':pilots,
        'teacher_combined_lower_bound_gpu_hours':teacher['combined_first_successful_batches_plus_resume_reservation_gpu_hours'],
        'training_reserved_gpu_hours':sum(x['gpu_hours'] for x in training),
        'full_eval_reserved_gpu_hours':sum(x['gpu_hours'] for x in evaluations)+sum(x['gpu_hours'] for x in failed_attempts),
        'conservative_all_eight_cards_interval':{'start_baseline_commit':'2ff4f41','start':start_text,
            'end':end.isoformat(),'gpu_hours_ceiling':ceiling,'under_192_gpu_hours':ceiling<=192},
        'scope':'Reservations include loading/recovery/waiting; not kernel utilization or billing. Detailed rows omit some early probes and interrupted teacher work, so are not an exact total. Whole-interval eight-card ceiling includes these omissions, idle/CPU time and pre-task time since the baseline commit.'}
(root/'final_cost.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ['training_reserved_gpu_hours','full_eval_reserved_gpu_hours','conservative_all_eight_cards_interval']},indent=2))

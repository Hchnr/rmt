"""Separate measured reservations from conservative active-window ceilings."""
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
for path in sorted(root.glob('full_eos_teacher32*_cost.json')):
    row=json.loads(path.read_text())
    if row['status']!='completed':raise ValueError('Failed evaluation requires separate attempt accounting')
    evaluations.append({'report':str(path),'gpu_hours':row['reserved_gpu_hours']})
if len(evaluations)!=6:raise ValueError('All six preregistered full tasks must complete first')
failed_attempts=[]
for path in sorted((Path('artifacts/v0.0.4_dynamic_recurr')/'eval_attempts').rglob('cost.json')):
    row=json.loads(path.read_text())
    if row['status']!='failed':raise ValueError(f'Unexpected non-failed archived attempt: {path}')
    failed_attempts.append({'report':str(path),'gpu_hours':row['reserved_gpu_hours']})
for path in sorted(root.glob('full_resume*_cost.json')):
    row=json.loads(path.read_text())
    if row['status']=='failed':
        failed_attempts.append({'report':str(path),'gpu_hours':row['reserved_gpu_hours']})
pilots=[]
for name in ['teacher_pilot','teacher_batch8_probe']:
    path=Path('artifacts/v0.0.4_dynamic_recurr')/name/'rank_0_summary.json'
    row=json.loads(path.read_text());pilots.append({'name':name,'gpu_hours':row['wall_seconds']/3600})
teacher=json.loads((root/'teacher_cost.json').read_text())
# The original run stopped after its last progress commit and resumed days later.
# Bound those active windows separately; do not charge the idle gap.
start_text=subprocess.check_output(['git','show','-s','--format=%cI','2ff4f41'],text=True).strip()
stop_text=subprocess.check_output(['git','show','-s','--format=%cI','018f371'],text=True).strip()
start=datetime.fromisoformat(start_text);interruption=datetime.fromisoformat(stop_text)
resume_reports=sorted(root.glob('full_resume*_transport.json'))
if not resume_reports:raise ValueError('Missing recorded resumed evaluation start')
recorded_resume_starts=[]
for path in resume_reports:
    row=json.loads(path.read_text())
    if row.get('started_at_utc'):
        recorded_resume_starts.append(datetime.fromisoformat(row['started_at_utc']))
# A checked-out report's filesystem mtime is not its launch time. Prefer the
# durable timestamp when available; retain mtime only for legacy reports.
resume_start=(min(recorded_resume_starts) if recorded_resume_starts else
              min(datetime.fromtimestamp(p.stat().st_mtime,timezone.utc) for p in resume_reports))
# Anchor the active-window ceiling to the last completed GPU job. Never use
# wall-clock "now": later user wait time would silently inflate the estimate.
activity=json.loads((root/'gpu_activity_window.json').read_text())
end=datetime.fromisoformat(activity['resume_end_utc'])
if not (start<interruption<resume_start<end):raise ValueError('Evaluation active windows are not chronological')
pre_ceiling=(interruption-start).total_seconds()*8/3600
resume_ceiling=(end-resume_start).total_seconds()*8/3600
ceiling=pre_ceiling+resume_ceiling
result={'as_of_utc':end.isoformat(),'known_training_reservations':training,
        'known_full_eval_reservations':evaluations,'failed_eval_attempts':failed_attempts,'teacher_pilots':pilots,
        'teacher_combined_lower_bound_gpu_hours':teacher['combined_first_successful_batches_plus_resume_reservation_gpu_hours'],
        'training_reserved_gpu_hours':sum(x['gpu_hours'] for x in training),
        'full_eval_reserved_gpu_hours':sum(x['gpu_hours'] for x in evaluations)+sum(x['gpu_hours'] for x in failed_attempts),
        'conservative_all_eight_cards_active_windows':{'pre_interruption':{
                'start_baseline_commit':'2ff4f41','start':start_text,'end_last_progress_commit':'018f371',
                'end':interruption.isoformat(),'gpu_hours_ceiling':pre_ceiling},
            'resume':{'start_first_resume_transport_report':resume_start.isoformat(),
                'end':end.isoformat(),'end_evidence':activity['end_evidence'],
                'gpu_hours_ceiling':resume_ceiling},
            'combined_gpu_hours_ceiling':ceiling,'under_192_gpu_hours':ceiling<=192},
        'scope':'Reservations include replica loading, recovery, evaluation waits and cleanup; not kernel utilization or billing. The eight-card ceilings conservatively charge all eight GPUs during the two documented active windows and exclude the four-day idle gap. Some early probes and interrupted work are not itemized; the ceiling bounds them.'}
(root/'final_cost.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ['training_reserved_gpu_hours','full_eval_reserved_gpu_hours','conservative_all_eight_cards_active_windows']},indent=2))

"""Report real training tokens, depth, timings and independently owned stage costs."""
import argparse
import json
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--reports', nargs='+', required=True)
p.add_argument('--output', required=True)
a=p.parse_args()
archives=[]
for path in Path('reports/v0.0.4_dynamic_recurr/training_stages').glob('*.json'):
    for row in json.loads(path.read_text()):
        archives.append({**row, 'archive': str(path)})
rows=[]
for path in a.reports:
    report=json.loads(Path(path).read_text());assert report['status']=='passed'
    world=report['identity']['world_size'];steps=report['steps']
    seconds=sum(x['seconds'] for x in steps)
    hist=[sum(x['exit_depth_counts'][i] for x in steps) for i in range(len(steps[0]['exit_depth_counts']))]
    inputs=sum(x['input_tokens'] for x in steps)
    assert inputs==sum(hist)
    stages=[x for x in archives if Path(x['report']).resolve()==Path(path).resolve()]
    if any(x['status']!='passed' for x in stages):raise ValueError('Training stage not fully restored/passed')
    stage_gpu_hours=sum(x['wall_seconds']*len(x['gpus'].split(','))/3600 for x in stages) if stages else None
    rows.append({'report':path,'world_size':world,'steps':len(steps),
        'input_tokens':inputs,'targets':sum(x['targets'] for x in steps),
        'mean_train_depth':sum(i*n for i,n in enumerate(hist))/sum(hist),
        'depth_histogram':hist,'timed_step_seconds':seconds,'timed_step_gpu_hours':seconds*world/3600,
        'post_load_wall_gpu_hours':report['wall_seconds']*world/3600,
        'owned_stage_gpu_hours_including_load_and_fresh_restore':stage_gpu_hours,
        'archives':stages,'peak_allocated_bytes':max(x['peak_allocated_bytes'] for x in steps),
        'maximum_off_prior_fraction':max(x['off_prior_fraction'] for x in steps),
        'compile':report['identity']['config'].get('compile',False),
        'scope':'Train depth includes forced-cap curriculum. Step timings exclude diagnostics/load/save; stage reservation includes independent restore. These cost columns overlap and must not be added.'})
Path(a.output).write_text(json.dumps({'rows':rows,'comparison':'Same teacher-data controls have matched tokens, not matched GPU-hours; no equal-training-compute superiority claim.'},indent=2)+'\n')
for row in rows:print(row['report'],row['input_tokens'],row['mean_train_depth'],row['timed_step_gpu_hours'])

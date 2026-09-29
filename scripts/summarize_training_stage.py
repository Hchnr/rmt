"""Summarize measured throughput and paired short-training holdout outcomes."""
import json
from pathlib import Path
import statistics
root=Path('reports/v0.0.4');out={'probes':{},'training':{}}
for p in sorted(root.glob('probe_*.json')):
 d=json.loads(p.read_text());rows=d['steps'][1:]
 seconds=sum(x['seconds'] for x in rows);tokens=sum(x['input_tokens'] for x in rows);targets=sum(x['targets'] for x in rows)
 out['probes'][p.stem]={'world_size':d['identity']['world_size'],'deterministic':d['identity']['config'].get('deterministic',False),'sequence_length':d['identity']['config']['sequence_length'],
  'hot_steps':len(rows),'input_tokens_per_second':tokens/seconds,'targets_per_second':targets/seconds,
  'peak_allocated_gib':max(x['peak_allocated_bytes'] for x in rows)/1024**3,
  'off_layer_fraction_mean':statistics.mean(x['off_layer_fraction'] for x in rows),
  '24h_hot_input_tokens_upper_bound':tokens/seconds*86400,
  'scope':'Excludes load/compile/eval/save; same teacher and exact KL included; no linear GPU scaling assumption'}
for name in ['fixed_128','opened_128']:
 p=root/(name+'.json')
 if not p.exists():continue
 d=json.loads(p.read_text());steps=d['steps'];out['training'][name]={'world_size':d['identity']['world_size'],
  'train_input_tokens':sum(x['input_tokens'] for x in steps),'train_targets':sum(x['targets'] for x in steps),
  'training_step_seconds':sum(x['seconds'] for x in steps),'post_load_wall_seconds':d['wall_seconds'],
  'deterministic':d['identity']['config'].get('deterministic',False),
  'initial_validation':d['validation'][0],'final_validation':d['validation'][-1],
  'max_train_off_layer_fraction':max(x['off_layer_fraction'] for x in steps),
  'hf_checkpoint':d['identity']['config']['artifact_dir']+'/hf'}
if len(out['training'])==2:
 f=out['training']['fixed_128'];l=out['training']['opened_128']
 assert f['train_input_tokens']==l['train_input_tokens'] and f['train_targets']==l['train_targets']
 out['comparison']={'same_training_token_budget':True,'opened_minus_fixed_dev_ce':l['final_validation']['ce']-f['final_validation']['ce'],
 'opened_over_fixed_training_time':l['training_step_seconds']/f['training_step_seconds'],
 'scope':'Same tokens and examples, not equal GPU-hours; development diagnostic, not benchmark superiority'}
(root/'training_summary.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))

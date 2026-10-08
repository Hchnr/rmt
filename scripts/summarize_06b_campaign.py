"""Summarize executed training, paired development evidence, and reservation costs."""
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

root = Path('reports/v0.0.5')
read = lambda name: json.loads((root / name).read_text())
reference = read('native_development512.json')
exposure = {Path(r['config']).stem: r for r in read('all_training_exposure.json')['rows']}
names = ['pilot_ce', 'pilot_kl01', 'pilot_kl05', 'short_kl05', 'mixed_kl05', 'mixed_kl05_lr1e6']
rows = []
for name in names:
    training = read(name + '.json')
    probe_file = name + ('_development.json' if name in names[:2] else '_development512.json')
    probe = read(probe_file)
    native = read('native_development.json') if name in names[:2] else reference
    assert native['cases_sha256'] == probe['cases_sha256']
    assert native['generation'] == probe['generation']
    by_id = {r['id']: r for r in native['rows']}
    domains = {}
    for domain in ['instruction', 'math']:
        selected = [r for r in probe['rows'] if r['domain'] == domain]
        delta = np.array([int(r['passed']) - int(by_id[r['id']]['passed']) for r in selected])
        indices = np.random.default_rng(17).integers(0, len(delta), (10000, len(delta)))
        domains[domain] = {'n': len(delta), 'score': probe['scores'][domain],
            'native_score': native['scores'][domain], 'paired_delta': float(delta.mean()),
            'paired_bootstrap_95pct': np.quantile(delta[indices].mean(1), [.025, .975]).tolist(),
            'passes_2pp_engineering_gate': bool(delta.mean() >= -.02)}
    e = exposure[name]
    rows.append({'name': name, 'initialization': 'untrained Qwen3-0.6B layer-order mapping',
        'recurrences': 28, 'completed_steps': training['completed_steps'],
        'targets': e['targets'], 'unique_source_ids_seen': e['unique_source_ids_seen'],
        'input_tokens': e['input_tokens'], 'output_budget': probe['generation']['max_new_tokens'],
        'domains': domains, 'passes_quality_gate': all(d['passes_2pp_engineering_gate'] for d in domains.values()),
        'train_report': str(root / (name + '.json')), 'probe_report': str(root / probe_file)})
stages = [json.loads(p.read_text()) for p in sorted((root / 'stages').glob('*.json'))]
staged_gpus = sorted({gpu for stage in stages for gpu in stage.get('gpus', [])})
assert set(staged_gpus) <= {'4', '5', '6', '7'}
cost_groups = {'training': 0., 'formal_evaluation': 0., 'development_and_verification': 0.}
for stage in stages:
    command = stage.get('command', [])
    if 'rmt.train' in command and '--verify-resume' not in command:
        category = 'training'
    elif 'scripts/evaluate_trained_checkpoint.py' in command:
        category = 'formal_evaluation'
    else:
        category = 'development_and_verification'
    cost_groups[category] += stage.get('reserved_gpu_hours', 0.)
now = datetime.now(timezone.utc)
running_cost = sum(max(0., (now-datetime.fromisoformat(s['started_at'])).total_seconds())*len(s['gpus'])/3600
                   for s in stages if s['status']=='running')
historical_passers = [r['name'] for r in rows if r['passes_quality_gate']]
recheck_path = root / 'deterministic_development_comparison.json'
recheck = json.loads(recheck_path.read_text()) if recheck_path.exists() else None
gate_passers = recheck['eligible_trained_recipes'] if recheck else historical_passers
result = {'generated_at': now.isoformat(), 'rows': rows,
    'recorded_original_protocol_gate_passers': historical_passers,
    'deterministic_recheck': str(recheck_path) if recheck else 'pending',
    'current_gate_passers': gate_passers,
    'decision': 'No trained recipe passes the available gate measurements.' if not gate_passers else 'Gate-passing trained recipes: '+', '.join(gate_passers),
    'uncertainty': 'Synthetic development probes only, 48 items per domain, one training seed. Bootstrap is conditional on these responses and does not estimate training-seed variance; all-equal outcomes give degenerate intervals.',
    'cost': {'completed_stage_reserved_gpu_hours': sum(s.get('reserved_gpu_hours', 0) for s in stages),
        'staged_jobs_physical_gpus': staged_gpus,
        'completed_reserved_gpu_hours_by_category': cost_groups,
        'running_reserved_gpu_hours_estimate': running_cost,
        'running_stages': [s['name'] for s in stages if s['status'] == 'running'],
        'scope': 'Stage reservation totals only; nested wrapper costs excluded to avoid double counting. Historical v0.0.4 teacher generation excluded; no new teacher generation.'}}
(root / 'campaign_summary.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result['cost'], indent=2))

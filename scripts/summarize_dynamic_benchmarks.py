"""Compare complete frozen students with the same-answer, seeded native baseline."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from rmt.evaluation.audit import audit

p = argparse.ArgumentParser()
p.add_argument('--runs', nargs='+', required=True)
p.add_argument('--output', required=True)
a = p.parse_args()
baseline = json.loads(Path('reports/v0.0.4/full_baseline.json').read_text())['results']
seeded = json.loads(Path('reports/v0.0.4_dynamic_recurr/ifeval_seeded_baseline.json').read_text())
assert seeded['status'] == 'passed' and seeded['seed'] == 17

def load(work, benchmark, native=False):
    work = Path(work)
    provenance = json.loads((work/'provenance.json').read_text())
    scores = {}
    for path in (work/'reviews').rglob('*.jsonl'):
        for row in map(json.loads, path.read_text().splitlines()):
            key = hashlib.sha256(row['input'].encode()).hexdigest()
            assert key not in scores
            sample = row['sample_score']
            scores[key] = (seeded['scores'][str(sample['sample_metadata']['key'])]
                           if native and benchmark == 'ifeval' else sample['score']['value'])
    identities = {(x['rmt_metadata']['prompt_sha256'], x['rmt_metadata']['seed'])
                  for x in map(json.loads, (work/'responses.jsonl').read_text().splitlines())}
    return scores, identities, provenance

native = {b: load(r['work'], b, True) for b, r in baseline.items()}
results = []
student_scores = {}
for run_path in a.runs:
    for run in json.loads(Path(run_path).read_text()):
        b = run['benchmark']; expected = {'ifeval': 541, 'math_500': 500}[b]
        checked = audit(run['work'], expected)
        responses=list(map(json.loads,(Path(run['work'])/'responses.jsonl').read_text().splitlines()))
        unique={(x['rmt_metadata']['prompt_sha256'],x['rmt_metadata']['seed']):x for x in responses}
        lengths=[x['usage']['completion_tokens'] for x in unique.values()]
        retries=[x['rmt_metadata'].get('batch_split_retries',0) for x in unique.values()]
        checked['output_length_tokens']={'p50':float(np.quantile(lengths,.5)),
            'p90':float(np.quantile(lengths,.9)),'p99':float(np.quantile(lengths,.99)),'max':max(lengths)}
        checked['batch_split_retries']={'affected_requests':sum(x>0 for x in retries),'max':max(retries)}
        recurrence=[x['rmt_metadata']['recurrence'] for x in unique.values() if x['rmt_metadata'].get('recurrence')]
        if recurrence:
            totals={key:sum(row[key] for row in recurrence) for key in recurrence[0]}
            request_means=[r['decode_depth_sum']/r['decode_positions'] for r in recurrence if r['decode_positions']]
            checked['recurrence']={'totals':totals,
                'prefill_token_weighted_mean':totals['prefill_depth_sum']/totals['prefill_positions'],
                'decode_token_weighted_mean':totals['decode_depth_sum']/max(1,totals['decode_positions']),
                'decode_request_weighted_mean':float(np.mean(request_means)) if request_means else None,
                'scope':'cap_positions denotes the configured execution limit; fixed36/40 limit exits are not hard48 dynamic caps.'}
        assert checked['unique_responses'] == expected
        scores, identities, provenance = load(run['work'], b)
        reference, native_ids, native_provenance = native[b]
        assert sorted(provenance['metadata']['generation_termination']['eos_token_ids']) == [151643,151645]
        assert provenance['metadata']['weights']['generation_config.json'] == native_provenance['metadata']['weights']['generation_config.json']
        student_scores[(run['candidate'], b)] = scores
        assert scores.keys() == reference.keys() and len(scores) == expected
        assert identities == native_ids
        assert provenance['protocol']['generation'] == native_provenance['protocol']['generation']
        assert provenance['selection'] == native_provenance['selection']
        if b == 'ifeval':
            assert provenance['scoring']['ifeval_score_seed'] == 17
        metric = 'prompt_level_strict' if b == 'ifeval' else 'acc'
        keys = sorted(scores)
        delta = np.array([scores[k][metric]-reference[k][metric] for k in keys])
        sample = np.random.default_rng(1704).integers(0, expected, (2000, expected))
        result = {**run, 'audit': checked, 'metric': metric,
                  'score': sum(scores[k][metric] for k in keys)/expected,
                  'native_score_same_scoring': sum(reference[k][metric] for k in keys)/expected,
                  'paired_delta': float(delta.mean()),
                  'paired_bootstrap_95pct': np.quantile(delta[sample].mean(1), [.025,.975]).tolist(),
                  'improved_questions': int((delta>0).sum()), 'worsened_questions': int((delta<0).sum()),
                  'identical_generation_selection_and_request_seeds': True}
        checked['uncertainty_note'] = 'Entire pinned benchmark, one response per prompt; intervals conditional on this generation seed and protocol.'
        results.append(result)
paired_controls = []
for b in ['ifeval', 'math_500']:
    candidates = {name: scores for (name, benchmark), scores in student_scores.items() if benchmark == b}
    for reference_part in ['fixed36', 'fixed40']:
        ref = [name for name in candidates if reference_part in name]
        dyn = [name for name in candidates if 'hybrid' in name]
        if len(ref) != 1 or len(dyn) != 1:
            raise ValueError(f'Expected one {reference_part} and hybrid control for {b}')
        reference, dynamic = candidates[ref[0]], candidates[dyn[0]]
        assert reference.keys() == dynamic.keys()
        metric = 'prompt_level_strict' if b == 'ifeval' else 'acc'
        delta = np.array([dynamic[k][metric]-reference[k][metric] for k in sorted(reference)])
        sample = np.random.default_rng(1704).integers(0, len(delta), (2000,len(delta)))
        paired_controls.append({'benchmark':b,'candidate':dyn[0],'reference':ref[0],
            'paired_delta':float(delta.mean()),'paired_bootstrap_95pct':np.quantile(delta[sample].mean(1),[.025,.975]).tolist(),
            'improved_questions':int((delta>0).sum()),'worsened_questions':int((delta<0).sum())})
Path(a.output).write_text(json.dumps({'rows': results, 'paired_student_controls':paired_controls,
    'scope': 'Official rules; per-question seeded IFEval for all models, native answers rescored without regeneration. Paired bootstrap conditions on one response per prompt; no paper protocol parity claim.'}, indent=2)+'\n')
for row in results:
    print(row['candidate'], row['benchmark'], row['score'], row['paired_delta'])

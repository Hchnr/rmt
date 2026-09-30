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
for run_path in a.runs:
    for run in json.loads(Path(run_path).read_text()):
        b = run['benchmark']; expected = {'ifeval': 541, 'math_500': 500}[b]
        checked = audit(run['work'], expected)
        assert checked['unique_responses'] == expected
        scores, identities, provenance = load(run['work'], b)
        reference, native_ids, native_provenance = native[b]
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
Path(a.output).write_text(json.dumps({'rows': results,
    'scope': 'Official rules; per-question seeded IFEval for all models, native answers rescored without regeneration. Paired bootstrap conditions on one response per prompt; no paper protocol parity claim.'}, indent=2)+'\n')
for row in results:
    print(row['candidate'], row['benchmark'], row['score'], row['paired_delta'])

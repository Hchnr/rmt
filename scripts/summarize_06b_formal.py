"""Audit complete frozen 0.6B runs and report same-prompt paired uncertainty."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from rmt.evaluation.audit import audit


def load(run):
    work = Path(run['work'])
    expected = {'ifeval': 541, 'math_500': 500}[run['benchmark']]
    checked = audit(work, expected)
    assert checked['unique_responses'] == expected
    provenance = json.loads((work / 'provenance.json').read_text())
    scores = {}
    metric = 'prompt_level_strict' if run['benchmark'] == 'ifeval' else 'acc'
    for path in sorted((work / 'reviews').rglob('*.jsonl')):
        for line in path.read_text().split('\n'):
            if not line.strip():
                continue
            row = json.loads(line)
            key = hashlib.sha256(row['input'].encode()).hexdigest()
            assert key not in scores
            scores[key] = row['sample_score']['score']['value'][metric]
    assert len(scores) == expected
    responses = [json.loads(line) for line in (work / 'responses.jsonl').read_text().split('\n') if line.strip()]
    identities = {(r['rmt_metadata']['prompt_sha256'], r['rmt_metadata']['seed']) for r in responses}
    return scores, identities, provenance, checked


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--native', required=True)
    p.add_argument('--candidate', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    native = {r['benchmark']: r for r in json.loads(Path(a.native).read_text())}
    candidate = {r['benchmark']: r for r in json.loads(Path(a.candidate).read_text())}
    assert set(native) == set(candidate) == {'math_500', 'ifeval'}
    results = []
    for benchmark in sorted(native):
        base, base_ids, bp, ba = load(native[benchmark])
        trained, trained_ids, tp, ta = load(candidate[benchmark])
        assert base.keys() == trained.keys() and base_ids == trained_ids
        assert bp['protocol']['generation'] == tp['protocol']['generation']
        assert bp['selection'] == tp['selection']
        assert bp['metadata']['weights']['generation_config.json'] == tp['metadata']['weights']['generation_config.json']
        if benchmark == 'ifeval':
            assert bp['scoring']['ifeval_score_seed'] == tp['scoring']['ifeval_score_seed'] == 17
        delta = np.array([trained[k] - base[k] for k in sorted(base)])
        indices = np.random.default_rng(17).integers(0, len(delta), (10000, len(delta)))
        results.append({'benchmark': benchmark, 'n': len(delta),
            'native_score': float(np.mean(list(base.values()))),
            'candidate_score': float(np.mean(list(trained.values()))),
            'paired_delta': float(delta.mean()),
            'paired_bootstrap_95pct': np.quantile(delta[indices].mean(1), [.025, .975]).tolist(),
            'improved': int((delta > 0).sum()), 'worsened': int((delta < 0).sum()),
            'native_audit': ba, 'candidate_audit': ta,
            'native_work': native[benchmark]['work'], 'candidate_work': candidate[benchmark]['work']})
    Path(a.output).write_text(json.dumps({'rows': results,
        'scope': 'Frozen best development candidate, fixed28, same generation protocol and request seeds. Full official EvalScope sets; paper prompt parity not established. Bootstrap conditions on one training seed and one response per prompt; no dynamic-depth benefit claim.'}, indent=2) + '\n')
    print(json.dumps([{k: r[k] for k in ['benchmark', 'n', 'native_score', 'candidate_score', 'paired_delta', 'paired_bootstrap_95pct']} for r in results], indent=2))


if __name__ == '__main__':
    main()

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
    unique = {(r['rmt_metadata']['prompt_sha256'], r['rmt_metadata']['seed']): r for r in responses}
    lengths = [r['usage']['completion_tokens'] for r in unique.values()]
    checked['completion_length'] = {'p50': float(np.quantile(lengths, .5)),
        'p90': float(np.quantile(lengths, .9)), 'p99': float(np.quantile(lengths, .99)), 'max': max(lengths)}
    retries = [r['rmt_metadata'].get('batch_split_retries', 0) for r in unique.values()]
    checked['batch_split_retries'] = {'affected_requests': sum(n > 0 for n in retries), 'max': max(retries)}
    recurrence = [r['rmt_metadata']['recurrence'] for r in unique.values()
                  if r['rmt_metadata'].get('recurrence', {}).get('prefill_positions', 0) > 0]
    if recurrence:
        totals = {key: sum(r[key] for r in recurrence) for key in recurrence[0]}
        checked['recurrence'] = {'totals': totals,
            'prefill_mean': totals['prefill_depth_sum'] / totals['prefill_positions'],
            'decode_mean': totals['decode_depth_sum'] / max(1, totals['decode_positions']),
            'scope': 'Fixed28 execution; cap_positions means its fixed execution limit, not a dynamic36 hard cap.'}
    checked['uncertainty_note'] = 'Entire pinned benchmark, one response per prompt; intervals conditional on this protocol and generation seed.'
    (work / 'audit.json').write_text(json.dumps(checked, indent=2) + '\n')
    return scores, identities, provenance, checked


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--native', '--reference', dest='native', required=True)
    p.add_argument('--reference-label', default='native')
    p.add_argument('--candidate', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--benchmarks', nargs='+', choices=['math_500', 'ifeval'], default=['math_500', 'ifeval'])
    a = p.parse_args()
    native = {r['benchmark']: r for r in json.loads(Path(a.native).read_text())}
    candidate = {r['benchmark']: r for r in json.loads(Path(a.candidate).read_text())}
    assert set(native) >= set(a.benchmarks) and set(candidate) >= set(a.benchmarks)
    results = []
    for benchmark in sorted(set(a.benchmarks)):
        base, base_ids, bp, ba = load(native[benchmark])
        trained, trained_ids, tp, ta = load(candidate[benchmark])
        assert base.keys() == trained.keys() and base_ids == trained_ids
        assert bp['protocol']['generation'] == tp['protocol']['generation']
        assert bp['selection'] == tp['selection']
        assert bp['metadata']['source_sha256'] == tp['metadata']['source_sha256']
        assert bp['metadata']['attention'] == tp['metadata']['attention']
        assert bp['metadata']['engine_environment'] == tp['metadata']['engine_environment']
        assert bp['metadata']['engine_environment']['deterministic_algorithms'] is True
        generation_configs = []
        for provenance in (bp, tp):
            data = (Path(provenance['metadata']['model']) / 'generation_config.json').read_bytes()
            assert hashlib.sha256(data).hexdigest() == provenance['metadata']['weights']['generation_config.json']
            config = json.loads(data)
            config.pop('transformers_version', None)
            generation_configs.append(config)
        assert generation_configs[0] == generation_configs[1]
        assert bp['metadata']['generation_termination'] == tp['metadata']['generation_termination']
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
            'paired_scores': [{'review_input_sha256': k, 'native': base[k], 'candidate': trained[k]}
                              for k in sorted(base)],
            'native_audit': ba, 'candidate_audit': ta,
            'native_work': native[benchmark]['work'], 'candidate_work': candidate[benchmark]['work']})
    if a.reference_label != 'native':
        for row in results:
            for suffix in ('score', 'audit', 'work'):
                row['reference_' + suffix] = row.pop('native_' + suffix)
            for pair in row['paired_scores']:
                pair['reference'] = pair.pop('native')
    Path(a.output).write_text(json.dumps({'rows': results, 'reference_label': a.reference_label, 'benchmarks': sorted(set(a.benchmarks)),
        'scope': 'Frozen fixed28 controls, same generation protocol and request seeds; reference_label identifies the baseline. Full selected official EvalScope sets; paper prompt parity not established. Bootstrap conditions on one training seed and one response per prompt; no dynamic-depth benefit claim.'}, indent=2) + '\n')
    print(json.dumps([{k: r[k] for k in ['benchmark', 'n', ('native_score' if a.reference_label == 'native' else 'reference_score'), 'candidate_score', 'paired_delta', 'paired_bootstrap_95pct']} for r in results], indent=2))


if __name__ == '__main__':
    main()

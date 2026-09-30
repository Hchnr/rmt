"""Summarize measured depth histograms and input-weighted document buckets."""
import argparse
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--reports', nargs='+', required=True)
p.add_argument('--output', required=True)
a = p.parse_args()

def histogram_summary(hist):
    total = sum(hist)
    if not total:
        return None
    def percentile(q):
        cumulative = 0
        for depth, count in enumerate(hist):
            cumulative += count
            if cumulative >= q * total:
                return depth
    return {'tokens': total, 'mean': sum(i*n for i,n in enumerate(hist))/total,
            'p50': percentile(.5), 'p90': percentile(.9), 'p99': percentile(.99),
            'max': max(i for i,n in enumerate(hist) if n),
            'fractions': {str(i): n/total for i,n in enumerate(hist) if n}}

results = []
for path in a.reports:
    report = json.loads(Path(path).read_text())
    corpus = {x['id']: x for x in map(json.loads, Path(report['data']).read_text().splitlines())}
    buckets = {}
    for doc in report['documents_detail']:
        row = corpus[doc['id']]
        length = len(row['input_ids'])
        length_bucket = '<=512' if length <= 512 else ('513-2048' if length <= 2048 else '>2048')
        for key in ('source:'+doc['domain'], 'length:'+length_bucket):
            b = buckets.setdefault(key, {'documents': 0, 'input_tokens': 0, 'depth_sum': 0., 'cap_sum': 0.})
            b['documents'] += 1; b['input_tokens'] += length
            b['depth_sum'] += length * doc['mean_depth']
            b['cap_sum'] += length * doc['cap_fraction']
    for b in buckets.values():
        b['mean_depth'] = b.pop('depth_sum') / b['input_tokens']
        b['cap_fraction'] = b.pop('cap_sum') / b['input_tokens']
    results.append({'report': path, 'all_input': histogram_summary(report['depth_histogram']),
                    'supervised_prediction_positions': histogram_summary(report['supervised_depth_histogram']),
                    'buckets': buckets})
Path(a.output).write_text(json.dumps({'rows': results,
    'scope': 'Observed teacher-forcing token depths. Source buckets are not semantic task labels; decode is reported separately.'}, indent=2)+'\n')

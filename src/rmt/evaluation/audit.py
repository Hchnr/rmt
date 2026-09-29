"""Validate completed official scorer outputs; never drop failed scores silently."""
import json
import math
from pathlib import Path


def wilson(success,total):
    if not total:return None
    z=1.959963984540054;p=success/total;den=1+z*z/total
    center=(p+z*z/(2*total))/den
    delta=z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/den
    return [max(0,center-delta),min(1,center+delta)]


def audit(work,expected):
    work=Path(work);rows=[];keys=set();values={}
    for file in sorted((work/'reviews').rglob('*.jsonl')):
        for line in file.read_text().splitlines():
            row=json.loads(line);key=(file.name,row['sample_score']['sample_id'])
            if key in keys:raise RuntimeError(f'Duplicate review {key}')
            keys.add(key);score=row['sample_score']['score'];value=score['value']
            if not value or score.get('metadata',{}).get('error'):raise RuntimeError(f'Invalid scorer result {key}')
            if str(score.get('explanation','')).startswith('Evaluation failed'):raise RuntimeError(f'Failed scorer {key}')
            for metric,x in value.items():
                if isinstance(x,(bool,int,float)):
                    if not math.isfinite(x):raise RuntimeError(f'Nonfinite score {key}')
                    values.setdefault(metric,[]).append(float(x))
            rows.append(row)
    if len(rows)!=expected:raise RuntimeError(f'Incomplete scores: expected {expected}, got {len(rows)}')
    responses=[]
    if (work/'responses.jsonl').exists():
        for line in (work/'responses.jsonl').read_text().splitlines():responses.append(json.loads(line))
    # Same input+seed has one identity. A replay must agree; transport retries
    # cannot inflate denominator or silently replace different model output.
    unique={}
    for response in responses:
        meta=response['rmt_metadata'];key=(meta['prompt_sha256'],meta['seed'])
        if key in unique and unique[key]['choices']!=response['choices']:raise RuntimeError('Nondeterministic duplicate response')
        unique[key]=response
    metrics={}
    for metric,xs in values.items():
        binary=all(x in (0,1) for x in xs)
        metrics[metric]={'mean':sum(xs)/len(xs),'n':len(xs),'wilson_95':wilson(sum(xs),len(xs)) if binary else None}
    out={'status':'passed','review_count':len(rows),'unique_responses':len(unique),'metrics':metrics,
         'length_fraction':sum(x['choices'][0]['finish_reason']=='length' for x in unique.values())/len(unique) if unique else None,
         'prompt_tokens':sum(x['usage']['prompt_tokens'] for x in unique.values()),
         'completion_tokens':sum(x['usage']['completion_tokens'] for x in unique.values()),
         'uncertainty_note':'Wilson intervals only for binary per-prompt metrics; quick fixed subsets, not unbiased full-benchmark estimates.'}
    (work/'audit.json').write_text(json.dumps(out,indent=2)+'\n');return out

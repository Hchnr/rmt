"""Compare frozen models under the same deterministic512-token probe protocol."""
import json
from pathlib import Path
import numpy as np

root=Path('reports/v0.0.5')
read=lambda name:json.loads((root/(name+'_development_deterministic512.json')).read_text())
base=read('native');reference={r['id']:r for r in base['rows']}
assert len(reference)==96
rows=[]
for name in ['untrained','pilot_ce','pilot_kl01','pilot_kl05','short_kl05','mixed_kl05','mixed_kl05_lr1e6']:
    report=read(name)
    assert report['cases_sha256']==base['cases_sha256'] and report['generation']==base['generation']
    assert report['attention']==base['attention']=='sdpa' and report['deterministic'] and base['deterministic']
    assert {r['id'] for r in report['rows']}==reference.keys() and len(report['rows'])==96
    domains={}
    for domain in ['instruction','math']:
        selected=[r for r in report['rows'] if r['domain']==domain]
        assert len(selected)==48
        delta=np.array([int(r['passed'])-int(reference[r['id']]['passed']) for r in selected])
        samples=np.random.default_rng(17).integers(0,48,(10000,48))
        domains[domain]={'n':48,'score':report['scores'][domain],'native_score':base['scores'][domain],
            'paired_delta':float(delta.mean()),'paired_bootstrap_95pct':np.quantile(delta[samples].mean(1),[.025,.975]).tolist(),
            'passes_2pp_gate':bool(delta.mean()>=-.02)}
    rows.append({'name':name,'domains':domains,'passes_quality_gate':all(x['passes_2pp_gate'] for x in domains.values())})
output={'rows':rows,'eligible_trained_recipes':[r['name'] for r in rows if r['name']!='untrained' and r['passes_quality_gate']],
    'formal_candidate_frozen_before_recheck':'mixed_kl05_lr1e6',
    'scope':'Verification of already frozen weights/questions/scorer after deterministic inference fixes; no new training or use of formal scores for selection. One seed and48 synthetic cases per domain, conditional bootstrap only; degenerate all-equal intervals are not population certainty.'}
(root/'deterministic_development_comparison.json').write_text(json.dumps(output,indent=2)+'\n')
print(json.dumps(output,indent=2))

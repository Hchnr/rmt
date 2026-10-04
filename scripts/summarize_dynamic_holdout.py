"""Paired document bootstrap for the exact same frozen held-out records."""
import argparse
import json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--reports',nargs='+',required=True);p.add_argument('--baseline',required=True);p.add_argument('--output',required=True)
a=p.parse_args();base=json.loads(Path(a.baseline).read_text());b={x['id']:x for x in base['documents_detail']};out=[]
for name in a.reports:
 r=json.loads(Path(name).read_text());d={x['id']:x for x in r['documents_detail']}
 if set(b)!=set(d) or r['data_sha256']!=base['data_sha256']:raise ValueError('Paired split mismatch')
 ids=sorted(b);weight=np.array([b[i]['targets'] for i in ids]);delta=np.array([d[i]['ce']-b[i]['ce'] for i in ids])
 if any(d[i]['targets']!=b[i]['targets'] for i in ids):raise ValueError('Paired target mismatch')
 rng=np.random.default_rng(1704);sample=rng.integers(0,len(ids),(2000,len(ids)))
 diffs=(delta[sample]*weight[sample]).sum(1)/weight[sample].sum(1)
 hist=np.array(r['depth_histogram']);total=hist.sum()
 out.append({'report':name,'ce':r['ce'],'mean_depth':r['mean_depth'],
   'delta_ce_vs_baseline':float((delta*weight).sum()/weight.sum()),'paired_document_bootstrap_95pct':np.quantile(diffs,[.025,.975]).tolist(),
   'fraction_at_depth40':float(hist[40]/total) if len(hist)>40 else 0.,'fraction_at_depth48':float(hist[48]/total) if len(hist)>48 else 0.,
   'domain_ce':r['domain_ce']})
result={'baseline':a.baseline,'data_sha256':base['data_sha256'],'rows':out,
 'scope':'2000 paired document bootstrap samples, seed 1704; conditional on one trained seed and these source documents; CE only, no task-score superiority claim.'}
Path(a.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(out,indent=2))

"""Paired quick-regression summaries of official EvalScope scores and raw responses."""
import argparse
import hashlib
import json
from pathlib import Path
from .audit import audit


def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',default='reports/v0.0.3/representative_runs.json')
    p.add_argument('--output',default='reports/v0.0.3/quality.json');a=p.parse_args()
    runs=json.loads(Path(a.runs).read_text());results={};pairs={}
    for run in runs:
        work=Path(run['work']);provenance=json.loads((work/'provenance.json').read_text())
        result=audit(work,provenance['selection']['count'])
        result['wall_seconds']=json.loads((work/'run_summary.json').read_text())['wall_seconds']
        result['work']=str(work);result['official_report']=json.loads(next((work/'reports').rglob('*.json')).read_text())
        results.setdefault(run['benchmark'],{})[run['model']]=result
        rows={}
        for path in (work/'reviews').rglob('*.jsonl'):
            for line in path.read_text().splitlines():
                row=json.loads(line);key=(path.name,row['sample_score']['sample_id'])
                rows[key]=row
        responses={}
        for line in (work/'responses.jsonl').read_text().splitlines():
            row=json.loads(line);responses[(row['rmt_metadata']['prompt_sha256'],row['rmt_metadata']['seed'])]=row
        pairs.setdefault(run['benchmark'],{})[run['model']]=(rows,responses)
    comparisons={}
    for name,pair in pairs.items():
        if set(pair)!= {'rmt','qwen'}:raise RuntimeError('Both backends required')
        rr,rs=pair['rmt'];qr,qs=pair['qwen']
        if rr.keys()!=qr.keys() or rs.keys()!=qs.keys():raise RuntimeError(f'Input/sample/seed mismatch: {name}')
        differences=[]
        for key in rr:
            if rr[key]['input']!=qr[key]['input']:raise RuntimeError('Different prompts')
            rv=rr[key]['sample_score']['score']['value'];qv=qr[key]['sample_score']['score']['value']
            if rv!=qv:differences.append({'subset_file':key[0],'sample_id':key[1],'rmt':rv,'qwen':qv,
                'input_sha256':hashlib.sha256(rr[key]['input'].encode()).hexdigest()})
        comparisons[name]={'same_prompt_and_seed':True,'n':len(rr),'score_differences':differences,
            'exact_text_agreement':sum(rs[k]['choices'][0]['message']['content']==qs[k]['choices'][0]['message']['content'] for k in rs)/len(rs)}
    output={'scope':'Fixed-sample non-thinking regression; per-run provenance/config records count and token limit. Not paper reproduction or evidence of superiority.',
        'results':results,'paired':comparisons,'gpu_hours_generation_and_scoring_upper_bound':sum(x['wall_seconds'] for group in results.values() for x in group.values())/3600}
    Path(a.output).write_text(json.dumps(output,indent=2,ensure_ascii=False)+'\n')
    for name,group in results.items():print(name,{model:r['official_report']['score'] for model,r in group.items()},comparisons[name]['exact_text_agreement'])


if __name__=='__main__':main()

"""Independent deterministic development probes, not official benchmark scores."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import re
from rmt.inference.runner import Runner, Generation


def cases():
    rows=[]
    topics=['rain','forests','bicycles','libraries','gardens','rivers','birds','clouds',
            'books','bridges','music','mountains','trains','oceans','flowers','snow']
    for i,topic in enumerate(topics):
        rows.append(dict(id=f'prefix_{i}',domain='instruction',kind='prefix',value='Note:',
            prompt=f'Write one sentence about {topic}. Start your response with the exact text "Note:".'))
        rows.append(dict(id=f'upper_{i}',domain='instruction',kind='upper',value='',
            prompt=f'Write a short sentence about {topic}. Use only uppercase letters for all words.'))
        rows.append(dict(id=f'json_{i}',domain='instruction',kind='json',value=topic,
            prompt=f'Return only a JSON object with exactly one key, "topic", whose value is "{topic}". Do not use a code fence.'))
    rng=random.Random(6005)
    for i in range(48):
        a,b,c=[rng.randrange(2,40) for _ in range(3)]
        rows.append(dict(id=f'arithmetic_{i}',domain='math',kind='integer',value=a*b+c,
            prompt=f'Calculate {a} * {b} + {c}. Reply with only the integer answer.'))
    return rows


def score(row,text):
    text=text.strip()
    if row['kind']=='prefix':return text.startswith(row['value'])
    if row['kind']=='upper':
        letters=[c for c in text if c.isalpha()]
        return bool(letters) and all('A'<=c<='Z' for c in letters)
    if row['kind']=='json':
        try:return json.loads(text)=={'topic':row['value']}
        except (ValueError,TypeError):return False
    return bool(re.fullmatch(r'[+-]?\d+',text)) and int(text)==row['value']


def correct(row,text):
    if row['kind']!='integer':return score(row,text)
    from math_verify import parse, verify
    return bool(verify(parse(str(row['value'])),parse(text)))


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True)
    p.add_argument('--backend',default='rmt',choices=['rmt','qwen'])
    p.add_argument('--output',required=True);p.add_argument('--compile',action='store_true')
    p.add_argument('--max-new-tokens',type=int,default=128)
    p.add_argument('--attention',choices=['eager','sdpa'],default='sdpa')
    p.add_argument('--deterministic',action='store_true')
    a=p.parse_args();items=cases()
    if a.deterministic:
        import torch
        os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.benchmark=False
    runner=Runner(a.model,backend=a.backend,compiled=a.compile,attention=a.attention,max_context=1024,
                  deterministic=a.deterministic)
    results=[]
    for start in range(0,len(items),4):
        batch=items[start:start+4]
        prompts=[runner.render([{'role':'user','content':r['prompt']}]) for r in batch]
        responses=runner.generate(prompts,[Generation(max_new_tokens=a.max_new_tokens,temperature=0,presence_penalty=0) for _ in batch])
        for row,response in zip(batch,responses):
            results.append(dict(row,response=response,passed=correct(row,response['text']),
                                format_passed=score(row,response['text'])))
        print(json.dumps({'completed':len(results),'total':len(items)}),flush=True)
    report={'model':a.model,'scoring_version':3,'math_scorer':'math-verify==0.8.0','cases_sha256':hashlib.sha256(json.dumps(items,sort_keys=True).encode()).hexdigest(),
        'attention':a.attention,'compiled':a.compile,'deterministic':a.deterministic,
        'generation':dict(max_new_tokens=a.max_new_tokens,temperature=0,presence_penalty=0,enable_thinking=False),
        'scores':{domain:sum(r['passed'] for r in results if r['domain']==domain)/sum(r['domain']==domain for r in results)
                  for domain in ('instruction','math')},'rows':results,
        'scope':'Synthetic formatting and arithmetic development diagnostics; not IFEval/MATH-500 and not broad quality evidence.'}
    Path(a.output).write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report['scores']),flush=True)


if __name__=='__main__':main()

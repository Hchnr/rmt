"""HTTP throughput of one versus four independent replicas, same request shape."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import statistics
import time
import requests


def request(item):
    model,port=item
    session=requests.Session();session.trust_env=False
    start=time.perf_counter()
    body={'model':model,'messages':[{'role':'user','content':'Write a detailed 500-word explanation of recurrent neural networks and their applications.'}],
        'max_tokens':32,'temperature':0,'presence_penalty':0,'seed':17}
    response=session.post(f'http://127.0.0.1:{port}/v1/chat/completions',json=body,timeout=300)
    response.raise_for_status();result=response.json()
    assert result['usage']['completion_tokens']==32 and result['choices'][0]['finish_reason']=='length'
    return time.perf_counter()-start,result['usage']['completion_tokens']


def batch(model,ports):
    start=time.perf_counter();jobs=[(model,port) for port in ports for _ in range(4)]
    with ThreadPoolExecutor(len(jobs)) as pool:results=list(pool.map(request,jobs))
    elapsed=time.perf_counter()-start
    return {'seconds':elapsed,'requests':len(jobs),'tokens':sum(x[1] for x in results),
        'tokens_per_second':sum(x[1] for x in results)/elapsed,'request_seconds':sorted(x[0] for x in results)}


out={}
for model,ports in [('rmt',[8901,8903,8905,8907]),('qwen',[8902,8904,8906,8908])]:
    rows={}
    for count in [1,4]:
        for _ in range(3):batch(model,ports[:count])
        runs=[batch(model,ports[:count]) for _ in range(3)]
        rows[str(count)]={'runs':runs,'median_tokens_per_second':statistics.median(x['tokens_per_second'] for x in runs)}
    out[model]={'replicas':rows,'four_vs_one':rows['4']['median_tokens_per_second']/rows['1']['median_tokens_per_second']}
Path('reports/v0.0.3/replica_scaling.json').write_text(json.dumps({'scope':'Four RMT and four Qwen single-GPU replicas on eight H100s; backend families measured separately, not an architecture-only comparison.','results':out},indent=2)+'\n')
print({k:v['four_vs_one'] for k,v in out.items()})

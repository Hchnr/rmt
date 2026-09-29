"""Live HTTP contract, independent seeded requests, stop, and thinking smoke."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import requests

p=argparse.ArgumentParser();p.add_argument('--rmt-port',type=int,default=8905);p.add_argument('--qwen-port',type=int,default=8906)
a=p.parse_args();results={}
def post(port,body):
    session=requests.Session();session.trust_env=False
    return session.post(f'http://127.0.0.1:{port}/v1/chat/completions',json=body,timeout=300)
for model,port in [('rmt',a.rmt_port),('qwen',a.qwen_port)]:
    base={'model':model,'messages':[{'role':'user','content':'What is 1 + 1?'}],'max_tokens':128,'temperature':0,'presence_penalty':0,'seed':17}
    errors=[]
    for change in [{'stream':True},{'n':2},{'model':'unknown'},{'max_tokens':0},{'max_tokens':38912},{'stop':['']}]:
        response=post(port,{**base,**change});assert response.status_code==400,(change,response.text)
        errors.append(change)
    with ThreadPoolExecutor(4) as pool:
        responses=[f.result() for f in [pool.submit(post,port,base) for _ in range(4)]]
    for response in responses:response.raise_for_status()
    bodies=[x.json() for x in responses]
    assert len({x['choices'][0]['message']['content'] for x in bodies})==1
    single=post(port,base);single.raise_for_status()
    text=single.json()['choices'][0]['message']['content'];assert text
    stopped=post(port,{**base,'stop':[text]});stopped.raise_for_status();stopped=stopped.json()
    assert stopped['choices'][0]['message']['content']=='' and stopped['choices'][0]['finish_reason']=='stop'
    thinking=post(port,{**base,'chat_template_kwargs':{'enable_thinking':True}});thinking.raise_for_status();thinking=thinking.json()
    assert thinking['rmt_metadata']['prompt_sha256']!=bodies[0]['rmt_metadata']['prompt_sha256']
    assert thinking['choices'][0]['message']['content']
    results[model]={'invalid_requests_rejected':len(errors),'concurrent_identical_requests':4,'stop_passed':True,
        'non_thinking':bodies[0],'thinking':thinking}
Path('reports/v0.0.3/service_contract.json').write_text(json.dumps({'status':'passed','backends':results},ensure_ascii=False,indent=2)+'\n')
print('service contract and thinking smoke passed')

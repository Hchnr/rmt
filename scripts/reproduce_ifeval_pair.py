"""Diagnostic only: replay the failed pair without modifying benchmark caches."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import threading
import time
import requests
from rmt.evaluation.protocol import request_seed
rows=[json.loads(x) for x in next(Path('artifacts/v0.0.4/datasets/full/ifeval').glob('*.jsonl')).open()]
barrier=threading.Barrier(2)
def call(index):
 messages=[{'role':'user','content':rows[index]['prompt']}]
 body={'model':'qwen','messages':messages,'max_tokens':32768,'temperature':.7,'top_p':.8,'top_k':20,
       'presence_penalty':1.5,'seed':request_seed(messages,17),'chat_template_kwargs':{'enable_thinking':False}}
 s=requests.Session();s.trust_env=False;barrier.wait();start=time.monotonic()
 r=s.post('http://127.0.0.1:9004/v1/chat/completions',json=body,timeout=7200)
 return {'index':index,'status_code':r.status_code,'seconds':time.monotonic()-start,'response':r.json(),'seed':body['seed']}
with ThreadPoolExecutor(2) as pool:results=list(pool.map(call,[484,485]))
Path('artifacts/v0.0.4/eval_recovery/ifeval_runtime/pair_diagnostic.json').write_text(json.dumps(results,indent=2)+'\n')
summary=[{k:v for k,v in x.items() if k!='response'}|{'usage':x['response'].get('usage'),'error':x['response'].get('error')} for x in results]
Path('reports/v0.0.4/ifeval_pair_diagnostic.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary),flush=True)

"""Schedule independent EvalScope tasks on explicitly supplied service replicas."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import requests


def run_one(model,port,benchmark,phase):
    session=requests.Session();session.trust_env=False
    for attempt in range(120):
        try:
            r=session.get(f'http://127.0.0.1:{port}/health',timeout=2);r.raise_for_status();break
        except requests.RequestException:time.sleep(1)
    else:raise RuntimeError(f'Service {port} did not become healthy')
    log=Path(f'reports/v0.0.3/{phase}_{model}_{benchmark}.log')
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','PYTHONPATH':'src'}
    cmd=[sys.executable,'-m','rmt.evaluation.run','--model',model,'--port',str(port),'--benchmark',benchmark,'--phase',phase]
    with log.open('w') as output:subprocess.run(cmd,env=env,stdout=output,stderr=subprocess.STDOUT,check=True)
    work=next(line.split('=',1)[1] for line in reversed(log.read_text().splitlines()) if line.startswith('EVAL_WORK_DIR='))
    print(model,benchmark,work,flush=True);return {'model':model,'benchmark':benchmark,'work':work}


def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',default='representative',choices=['pilot','representative'])
    p.add_argument('--rmt-ports',default='8801,8803,8805');p.add_argument('--qwen-ports',default='8802,8804,8806')
    p.add_argument('--benchmarks',default='live_code_bench,math_500,ifeval,mmlu_redux,ceval');a=p.parse_args()
    benchmarks=a.benchmarks.split(',');jobs=[]
    for model,ports in [('rmt',a.rmt_ports),('qwen',a.qwen_ports)]:
        for i,port in enumerate(map(int,ports.split(','))):jobs.append((model,port,benchmarks[i::len(ports.split(','))]))
    def worker(job):
        model,port,names=job;return [run_one(model,port,name,a.phase) for name in names]
    with ThreadPoolExecutor(len(jobs)) as pool:results=[entry for values in pool.map(worker,jobs) for entry in values]
    Path(f'reports/v0.0.3/{a.phase}_runs.json').write_text(json.dumps(results,indent=2)+'\n')


if __name__=='__main__':main()

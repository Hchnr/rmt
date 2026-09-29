"""Use EvalScope's LCB scorer with an isolated OS-restricted worker."""
import json
import os
from pathlib import Path
import subprocess
import sys


FAILURES=[]

def isolated_check(sample,generation,timeout,debug=False):
    worker=Path(__file__).with_name('code_worker.py')
    env={k:v for k,v in os.environ.items() if k in ['PATH','LD_LIBRARY_PATH','PYTHONPATH','HOME']}
    env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    count=len(json.loads(sample['input_output'])['inputs'])
    try:
        output=subprocess.run([sys.executable,str(worker)],input=json.dumps({'sample':sample,'generation':generation,'timeout':timeout}),
            text=True,capture_output=True,env=env,timeout=min(150,(timeout+1)*count+30),check=True)
        result=json.loads(output.stdout.strip().splitlines()[-1])
        return result['result'],result['metadata']
    except subprocess.TimeoutExpired:return [-1]*count,{'error':'sandbox wall timeout'}
    except subprocess.CalledProcessError as error:
        # Infrastructure errors must abort evaluation, never become wrong answers.
        FAILURES.append(error.stderr[-1500:])
        raise RuntimeError('LCB sandbox worker failed: '+error.stderr[-1500:]) from error


def install():
    from evalscope.benchmarks.live_code_bench import evaluate_utils
    evaluate_utils.codegen_check_correctness=isolated_check

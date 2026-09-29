"""Exercise the actual OS-restricted EvalScope code runner before using LCB."""
import json
from pathlib import Path
import subprocess
import sys
from rmt.evaluation.sandbox import isolated_check
worker=Path('src/rmt/evaluation/code_worker.py')
p=subprocess.run([sys.executable,str(worker)],input='{"probe":true}',text=True,capture_output=True,check=True)
probe=json.loads(p.stdout.strip().splitlines()[-1]);assert set(probe['denied'])=={'file','network','write'}
sample={'input_output':json.dumps({'inputs':['1 2\n','3 4\n'],'outputs':['3\n','7\n'],'fn_name':None})}
correct=isolated_check(sample,'a,b=map(int,input().split()); print(a+b)',2)[0]
wrong=isolated_check(sample,'print(0)',2)[0]
timeout=isolated_check(sample,'while True: pass',1)[0]
assert correct==[True,True] and not all(x is True for x in wrong) and not all(x is True for x in timeout)
out={'status':'passed','os_probe':probe,'correct':correct,'wrong':wrong,'timeout':timeout}
Path('reports/v0.0.3/sandbox.json').write_text(json.dumps(out,indent=2)+'\n');print(out)

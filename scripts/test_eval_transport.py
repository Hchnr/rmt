"""Verify deadline changes only queue waiting and preserves worker outputs/errors."""
import queue
import time
from serve_eval_transport import configure_timeout
from rmt.inference.server import Batcher
class Runner:
    def generate(self,prompts,configs):
        time.sleep(.08)
        return [{'text':prompt} for prompt in prompts]
configure_timeout(.01)
b=Batcher(Runner(),size=1,wait=0)
try:b.submit('late',None)
except queue.Empty:pass
else:raise AssertionError('Short deadline did not expire')
configure_timeout(.5)
assert b.submit('same output',None)=={'text':'same output','batch_split_retries':0}
for invalid in [0,-1,float('inf'),float('nan')]:
    try:configure_timeout(invalid)
    except ValueError:pass
    else:raise AssertionError('Invalid deadline accepted')
print('PASS: short deadline expires; extended waiting preserves exact result; invalid deadlines rejected')

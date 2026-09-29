"""Log progress and tracebacks around frozen inference, without changing outputs."""
import json
import runpy
import time
import traceback
from rmt.inference.runner import Runner
original_generate=Runner.generate

def generate(self,*args,**kwargs):
    started=time.monotonic();count=0
    def progress(module,inputs,kw):
        nonlocal count
        count+=1
        if count==1 or count%1000==0:
            print(json.dumps({'event':'decode_progress','step':count,'seconds':time.monotonic()-started,
                              'mask_shape':list(kw['attention_mask'].shape)}),flush=True)
    hook=self.model.register_forward_pre_hook(progress,with_kwargs=True)
    try:
        value=original_generate(self,*args,**kwargs)
        print(json.dumps({'event':'decode_finished','steps':count,'seconds':time.monotonic()-started}),flush=True)
        return value
    except Exception:
        print(json.dumps({'event':'decode_failed','steps':count,'seconds':time.monotonic()-started}),flush=True)
        traceback.print_exc()
        raise
    finally:hook.remove()
Runner.generate=generate
runpy.run_module('rmt.inference.server',run_name='__main__')

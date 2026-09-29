import queue
import torch
from rmt.inference.server import Batcher
from rmt.inference.runner import Generation


def test_oom_split_retains_config_and_response_identity(monkeypatch):
    class Fake:
        def generate(self,prompts,configs):
            if len(prompts)>1:raise torch.cuda.OutOfMemoryError('injected capacity failure')
            return [{'prompt':prompts[0],'seed':configs[0].seed}]
    batcher=Batcher.__new__(Batcher);batcher.runner=Fake()
    monkeypatch.setattr(torch.cuda,'empty_cache',lambda:None)
    jobs=[('one',Generation(seed=1),queue.Queue()),('two',Generation(seed=2),queue.Queue())]
    batcher.execute(jobs)
    assert jobs[0][2].get_nowait()=={'prompt':'one','seed':1}
    assert jobs[1][2].get_nowait()=={'prompt':'two','seed':2}

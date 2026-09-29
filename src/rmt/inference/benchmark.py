"""Real-weight eager/compiled cached inference comparison, no speed claims by construction."""
import argparse
import json
import statistics
import time
from pathlib import Path
import torch
from .runner import Runner,Generation
from ..cache import RmtCapacityCache
from ..runtime import write_report,versions


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',default='artifacts/bootstrap/rmt-bound-4b')
    p.add_argument('--output',default='reports/v0.0.3/performance.json');p.add_argument('--mixed',action='store_true');p.add_argument('--matrix',action='store_true')
    args=p.parse_args()
    torch.manual_seed(17)
    r=Runner(args.model,compiled=False,max_context=4096)
    if args.mixed:
        r.model.config.routing_mode='learned';r.model.model.cell.router.prior_strength=0
        with torch.no_grad():r.model.model.cell.router.weight.normal_(std=.01)
    texts=['Explain in one sentence why the sky is blue.','计算17加25，直接给出答案。',
           'Write a short Python function to add two numbers.','Name the capital of France.']
    prompts=[r.render([{'role':'user','content':s}]) for s in texts]
    cfg=Generation(max_new_tokens=32,temperature=0,presence_penalty=0)
    # Common teacher-forced tokens isolate numeric differences from sampling.
    ids=torch.tensor([[11,12,13,14,15,16]],device=r.device)
    with torch.inference_mode():
        a=r.model(ids,use_cache=False).logits
        cache=RmtCapacityCache(16);pieces=[]
        for start,end in [(0,3),(3,4),(4,6)]:
            pieces.append(r.model(ids[:,start:end],use_cache=True,past_key_values=cache).logits)
        torch.testing.assert_close(a,torch.cat(pieces,1),atol=.25,rtol=.02)
    cases=[(1,prompts[:1]),(4,prompts[:4])]
    if args.matrix:
        r.eos=set()  # Fixed output length, explicitly recorded below.
        cases=[(b,['test '*length]*b) for length,b in [(128,1),(512,4),(2048,1)]]
    eager=[];expected=[]
    for batch,case_prompts in cases:
        r.generate(case_prompts,[cfg]*batch)
        runs=[r.generate(case_prompts,[cfg]*batch) for _ in range(3)]
        eager.append(statistics.median(x[0]['batch_seconds'] for x in runs));expected.append(runs[-1])
    t=time.perf_counter();r.model.model.cell.bank.compile_projections()
    r.generate(prompts,[cfg]*4);cold=time.perf_counter()-t
    with torch.inference_mode():
        b=r.model(ids,use_cache=False).logits
        metric={'max_abs':(a.float()-b.float()).abs().max().item(),
                'relative_rmse':((a.float()-b.float()).square().mean().sqrt()/a.float().square().mean().sqrt()).item()}
        labels=ids[:,1:].reshape(-1)
        f=torch.nn.functional.cross_entropy
        delta=abs((f(a[:,:-1].float().reshape(-1,a.shape[-1]),labels)-f(b[:,:-1].float().reshape(-1,b.shape[-1]),labels)).item())
    rows=[]
    for j,(batch,case_prompts) in enumerate(cases):
        runs=[r.generate(case_prompts,[cfg]*batch) for _ in range(3)]
        compiled=statistics.median(x[0]['batch_seconds'] for x in runs)
        rows.append({'batch':batch,'prompt_tokens':runs[-1][0]['prompt_tokens'],'eager_seconds':eager[j],'compiled_seconds':compiled,'speedup':eager[j]/compiled,
                     'generated_tokens_per_second':sum(x['completion_tokens'] for x in runs[-1])/compiled,
                     'greedy_equal':all(x['token_ids']==y['token_ids'] for x,y in zip(expected[j],runs[-1]))})
    from torch._dynamo.utils import counters,compile_times
    status='passed' if metric['relative_rmse']<=1e-3 and delta<=1e-3 else 'failed_numeric'
    report={'status':status,'environment':versions(),'mixed':args.mixed,'matrix':args.matrix,'fixed_output_tokens':32 if args.matrix else None,'numeric':metric,'delta_nll':delta,
            'cold_compile_seconds':cold,'rows':rows,'peak_memory_bytes':torch.cuda.max_memory_allocated(),
            'compile_counters':{str(k):dict(v) for k,v in counters.items()},'compile_times':compile_times(repr='str')}
    write_report(args.output,report);print(json.dumps(report),flush=True)
    if status!='passed':raise RuntimeError('Compiled numerics outside fixed acceptance thresholds')


if __name__=='__main__':main()

"""Measure warmed cached generation, token identity and compilation on real exports."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import statistics
import time
import torch
from torch._dynamo.utils import counters
from rmt.inference.runner import Runner, Generation


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--output',required=True)
    p.add_argument('--tokens',type=int,default=32);p.add_argument('--repeats',type=int,default=3)
    p.add_argument('--batch-sizes',type=int,nargs='+',default=[1,4])
    p.add_argument('--prompt-repeats',type=int,default=1)
    a=p.parse_args();rows=[];reference={}
    if not a.batch_sizes or any(n<1 for n in a.batch_sizes) or len(set(a.batch_sizes))!=len(a.batch_sizes) or a.prompt_repeats<1:
        raise ValueError('Positive distinct batch sizes and positive prompt repeats required')
    for compiled in [False,True]:
        counters.clear();runner=Runner(a.model,compiled=compiled,attention='sdpa',max_context=4096)
        for size in a.batch_sizes:
            prompts=[runner.render([{'role':'user','content':' '.join([f'Explain in detail how to test a sorting algorithm. Include example {i+1}.']*a.prompt_repeats)}]) for i in range(size)]
            cfg=[Generation(max_new_tokens=a.tokens,temperature=0,presence_penalty=0) for _ in prompts]
            start=time.monotonic();warm=runner.generate(prompts,cfg);warm_seconds=time.monotonic()-start
            torch.cuda.reset_peak_memory_stats();measurements=[]
            for _ in range(a.repeats):
                result=runner.generate(prompts,cfg)
                measurements.append({'seconds':result[0]['batch_seconds'],'ttft':result[0]['ttft_seconds'],
                    'tokens':sum(x['completion_tokens'] for x in result)})
            tokens=[x['token_ids'] for x in result]
            if not compiled:reference[size]=tokens
            rows.append({'compiled':compiled,'batch_size':size,'warmup_seconds':warm_seconds,
                'prompt_sha256':[hashlib.sha256(p.encode()).hexdigest() for p in prompts],
                'prompt_tokens':[len(runner.tokenizer.encode(p,add_special_tokens=False)) for p in prompts],
                'median_seconds':statistics.median(x['seconds'] for x in measurements),
                'median_tokens_per_second':statistics.median(x['tokens']/x['seconds'] for x in measurements),
                'median_ttft_seconds':statistics.median(x['ttft'] for x in measurements),
                'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'greedy_tokens_match_eager':tokens==reference[size],
                'measurements':measurements,'recurrence':[x['recurrence'] for x in result],
                'compile_counters':{k:dict(v) for k,v in counters.items()},'token_ids':tokens})
            print(json.dumps({k:v for k,v in rows[-1].items() if k not in ['token_ids','compile_counters','recurrence']}),flush=True)
        del runner;gc.collect();torch.cuda.empty_cache();torch._dynamo.reset()
    report={'model':a.model,'output_budget':a.tokens,'repeats':a.repeats,'prompt_repeats':a.prompt_repeats,'rows':rows,
        'scope':'Projection compilation with eager dynamic control and rectangular KV; measured generation includes routing, stopping and sampling'}
    report['speedup_by_batch']={str(n):rows[i]['median_seconds']/rows[i+len(a.batch_sizes)]['median_seconds'] for i,n in enumerate(a.batch_sizes)}
    Path(a.output).write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()

"""Small real-model profile and cached/uncached production-path comparison."""
import json
from pathlib import Path
import statistics
import torch
from rmt.inference.runner import Runner,Generation
from rmt.runtime import versions
r=Runner('artifacts/bootstrap/rmt-bound-4b');r.eos=set()
cfg=Generation(max_new_tokens=16,temperature=0,presence_penalty=0)
rows=[]
for length in [128,512,2048]:
    prompts=['test '*length]
    row={'prompt_tokens':len(r.tokenizer.encode(prompts[0],add_special_tokens=False))}
    outputs={}
    for cached in [True,False]:
        for _ in range(3):r.generate(prompts,[cfg],use_cache=cached)
        runs=[r.generate(prompts,[cfg],use_cache=cached)[0] for _ in range(3)]
        key='cached' if cached else 'uncached'
        row[key+'_seconds']=statistics.median(x['batch_seconds'] for x in runs)
        outputs[key]=runs[-1]
    row['speedup']=row['uncached_seconds']/row['cached_seconds']
    row['greedy_equal']=outputs['cached']['token_ids']==outputs['uncached']['token_ids']
    rows.append(row)
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
    r.generate(['test '*128],[cfg])
report={'environment':versions(),'rows':rows,'fixed_generated_tokens':16,'eos_disabled':True,
    'scope':'Production cached/decode-replay vs uncached/compiled-prefill, not isolated KV FLOP savings.',
    'profile_cpu':prof.key_averages().table(sort_by='self_cpu_time_total',row_limit=20),
    'profile_gpu':prof.key_averages().table(sort_by='self_cuda_time_total',row_limit=20),
    'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()}
Path('reports/v0.0.3/cache_and_profile.json').write_text(json.dumps(report,indent=2)+'\n');print([(x['prompt_tokens'],x['speedup']) for x in rows],flush=True)

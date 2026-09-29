"""Offline batched generation from local HF checkpoints."""
import argparse
import json
from rmt.inference.runner import Runner,Generation
p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--backend',choices=['rmt','qwen'],default='rmt')
p.add_argument('--prompt',action='append',required=True);p.add_argument('--eager',action='store_true');p.add_argument('--thinking',action='store_true')
p.add_argument('--max-new-tokens',type=int,default=512);p.add_argument('--max-context',type=int,default=8192)
p.add_argument('--temperature',type=float,default=.7);p.add_argument('--seed',type=int,default=17)
a=p.parse_args();runner=Runner(a.model,a.backend,not a.eager,max_context=a.max_context)
cfg=Generation(max_new_tokens=a.max_new_tokens,temperature=a.temperature,seed=a.seed,enable_thinking=a.thinking)
prompts=[runner.render([{'role':'user','content':text}],a.thinking) for text in a.prompt]
print(json.dumps(runner.generate(prompts,[cfg]*len(prompts)),ensure_ascii=False,indent=2))

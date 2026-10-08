"""Replay identical long-prefix decode inputs through eager and compiled paths."""
import argparse
import json
import os
from pathlib import Path
import torch
from rmt.inference.runner import Runner
from rmt.cache import RmtCapacityCache


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default='reports/v0.0.5/compiled_decode_numeric.json')
    p.add_argument('--eager-qkv',action='store_true');p.add_argument('--eager-output',action='store_true')
    p.add_argument('--eager-prefill',action='store_true')
    p.add_argument('--no-compile',action='store_true')
    p.add_argument('--deterministic',action='store_true')
    a=p.parse_args()
    if a.deterministic:
        os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.benchmark=False
    runner=Runner('artifacts/v0.0.5/reference_hf',compiled=False,attention='sdpa',max_context=4096,
                  deterministic=a.deterministic)
    traces={}
    with torch.inference_mode():
        for size in (1,4,8):
            prompts=[runner.render([{'role':'user','content':' '.join([
                f'Explain in detail how to test a sorting algorithm. Include example {i+1}.']*32)}]) for i in range(size)]
            ids=runner.tokenizer(prompts,padding=True,return_tensors='pt').input_ids.cuda()
            cache=RmtCapacityCache(ids.shape[1]+32);trace=[]
            for step in range(32):
                mask=torch.ones(size,cache.get_seq_length()+ids.shape[1],device='cuda',dtype=torch.long)
                out=runner.model(ids,attention_mask=mask,past_key_values=cache,use_cache=True,logits_to_keep=1)
                trace.append((ids.cpu(),out.logits.float().cpu()))
                ids=out.logits[:,-1].argmax(-1,keepdim=True)
            traces[size]=trace
            del cache
        if not a.no_compile:runner.model.model.cell.bank.compile_projections()
        from rmt.experts import project_qkv,project_output
        if a.eager_qkv:runner.model.model.cell.bank._project_qkv=project_qkv
        if a.eager_output:runner.model.model.cell.bank._project_output=project_output
        if a.eager_prefill:
            runner.model.model.cell.bank._prefill_qkv=project_qkv
            runner.model.model.cell.bank._prefill_output=project_output
        rows=[]
        for size,trace in traces.items():
            cache=RmtCapacityCache(trace[0][0].shape[1]+32)
            for step,(fixed,expected) in enumerate(trace):
                ids=fixed.cuda();mask=torch.ones(size,cache.get_seq_length()+ids.shape[1],device='cuda',dtype=torch.long)
                actual=runner.model(ids,attention_mask=mask,past_key_values=cache,use_cache=True,logits_to_keep=1).logits.float().cpu()
                delta=actual-expected
                rows.append({'batch_size':size,'step':step,'max_abs':delta.abs().max().item(),
                    'relative_rmse':(delta.square().mean().sqrt()/expected.square().mean().sqrt()).item(),
                    'argmax_disagreements':(actual.argmax(-1)!=expected.argmax(-1)).sum().item()})
            del cache
    report={'status':'passed' if all(x['max_abs']<=.5 and x['relative_rmse']<.01 for x in rows) else 'failed',
            'eager_qkv':a.eager_qkv,'eager_output':a.eager_output,
            'eager_prefill':a.eager_prefill,
            'compiled':not a.no_compile,
            'deterministic':a.deterministic,
            'rows':rows,'scope':'Aligned teacher-forced replay of eager greedy continuation; separates numeric differences from free-running divergence.'}
    Path(a.output).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='rows'}),flush=True)
    assert report['status']=='passed'


if __name__=='__main__':main()

"""Real 4B forced-depth cache, HF export and compiled inference checks."""
import argparse
import json
from pathlib import Path
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from rmt.checkpoint import load_qwen_as_rmt, export_hf
from rmt.cache import RmtCapacityCache


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--base',default='/share/project/eai_pwm/models/Qwen/Qwen3-4B')
    p.add_argument('--max-depth',type=int,default=48)
    p.add_argument('--output-root',default='artifacts/v0.0.4_dynamic_recurr/reference_hf')
    p.add_argument('--report',default='reports/v0.0.4_dynamic_recurr/model_verification.json')
    a=p.parse_args()
    torch.set_num_threads(4)
    source=a.base
    model,_=load_qwen_as_rmt(source,max_recurrences=a.max_depth,halting_policy='hidden',
        recurrence_schedule='tail',attn_implementation='sdpa')
    model=model.cuda().eval()
    tok=AutoTokenizer.from_pretrained(source);ids=tok('Explain why recurrent attention must retain historical keys and values.',return_tensors='pt').input_ids.cuda()
    base_depth=model.config.num_experts
    stops=base_depth+torch.arange(ids.numel(),device='cuda').reshape_as(ids)%(a.max_depth-base_depth+1)
    report={'base':source,'base_depth':base_depth,'max_depth':a.max_depth,
            'scope':'tested real model shapes; numerical tolerances, not universal bitwise equality','checks':{}}
    with torch.inference_mode():
        eager=model(ids,forced_exit_depths=stops,use_cache=False)
        cache=RmtCapacityCache(ids.shape[1]+8);chunks=[]
        for t in range(ids.shape[1]):
            out=model(ids[:,t:t+1],forced_exit_depths=stops[:,t:t+1],past_key_values=cache,use_cache=True)
            chunks.append(out.logits)
        actual=torch.cat(chunks,1)
        def diff(a,b):
            delta=(a.float()-b.float())
            return {'max_abs':delta.abs().max().item(),'relative_rmse':(delta.square().mean().sqrt()/a.float().square().mean().sqrt()).item()}
        d=diff(eager.logits,actual);report['checks']['cached_vs_full']=d
        assert d['max_abs']<=.5 and d['relative_rmse']<.01,d
        assert all(cache.get_seq_length(i)==ids.shape[1] for i in range(a.max_depth))
        model.model.cell.bank.compile_projections()
        started=time.monotonic();compiled=model(ids,forced_exit_depths=stops,use_cache=False)
        torch.cuda.synchronize();report['compile_first_seconds']=time.monotonic()-started
        d=diff(eager.logits,compiled.logits);report['checks']['compiled_vs_eager']=d
        assert d['max_abs']<=.5 and d['relative_rmse']<.01,d
        assert torch.equal(compiled.exit_depths,stops)
        root=Path(a.output_root)
        export_hf(model,root,source)
        reference=eager.logits.cpu()
        del eager,compiled,model,cache;torch.cuda.empty_cache()
        reloaded=AutoModelForCausalLM.from_pretrained(root,trust_remote_code=True,local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval()
        restored=reloaded(ids,forced_exit_depths=stops,use_cache=False)
        d=diff(reference,restored.logits.cpu());report['checks']['hf_remote_code_reload']=d
        assert d['max_abs']==0,d
        report['parameters']=sum(p.numel() for p in reloaded.parameters())
    report['status']='passed'
    Path(a.report).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()

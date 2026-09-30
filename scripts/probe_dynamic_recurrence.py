"""Bounded real-model depth/halting diagnostic; never claims oracle routing gains."""
import argparse
import json
from pathlib import Path
import time
import torch
from rmt.checkpoint import load_qwen_as_rmt
from rmt.modeling_rmt import RmtForCausalLM
from rmt.training_data import load_packs


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model',default='/share/project/eai_pwm/models/Qwen/Qwen3-4B')
    ap.add_argument('--data',default='artifacts/v0.0.4/corpus/dev_pilot.jsonl')
    ap.add_argument('--output',default='reports/v0.0.4_dynamic_recurr/depth_probe.json')
    ap.add_argument('--length',type=int,default=512)
    ap.add_argument('--packs',type=int,default=2)
    ap.add_argument('--hf',action='store_true')
    ap.add_argument('--schedule',choices=['cycle','tail'],default=None)
    a=ap.parse_args();torch.set_num_threads(4)
    if a.hf:
        m=RmtForCausalLM.from_pretrained(a.model,dtype=torch.bfloat16,attn_implementation='sdpa')
    else:
        m,_=load_qwen_as_rmt(a.model,max_recurrences=48,attn_implementation='sdpa')
    m=m.cuda().eval();c=m.config
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
    packs=load_packs(a.data,a.length,tokenizer.pad_token_id,shuffle=False)[:a.packs]
    rows=[]
    with torch.inference_mode():
        for schedule in ([a.schedule] if a.schedule else ['cycle','tail']):
            c.recurrence_schedule=schedule
            settings=[('fixed',r,0.) for r in [36,40,48]]
            settings += [('hidden',48,t) for t in [.03,.06,.1,.2,.4,.8]]
            settings += [('probability',48,t) for t in [.001,.01,.03,.1,.3]]
            for policy,limit,tau in settings:
                c.halt_threshold=tau;c.halt_relative_threshold=tau;c.halt_probability_threshold=tau
                sums=torch.zeros(3,device='cuda');hist=torch.zeros(49,dtype=torch.long,device='cuda')
                torch.cuda.synchronize();start=time.monotonic()
                for batch in packs:
                    batch={k:v.cuda() for k,v in batch.items()}
                    out=m(**batch,use_cache=False,halting_policy=policy,recurrence_limit=limit,loss_chunk_size=64)
                    from rmt.losses import shifted_targets
                    n=(shifted_targets(batch['labels'],batch['attention_mask'],batch['segment_ids'])!=-100).sum()
                    valid=batch['attention_mask'].bool();d=out.exit_depths[valid]
                    sums+=torch.stack((out.ce_loss*n,n.float(),d.sum().float()))
                    hist+=torch.bincount(d,minlength=49)
                torch.cuda.synchronize()
                row=dict(schedule=schedule,policy=policy,limit=limit,threshold=tau,ce=(sums[0]/sums[1]).item(),
                    targets=sums[1].item(),mean_depth=(sums[2]/hist.sum()).item(),histogram=hist.tolist(),
                    seconds=time.monotonic()-start)
                rows.append(row);print(json.dumps(row),flush=True)
                Path(a.output).write_text(json.dumps({'model':a.model,'data':a.data,'length':a.length,'packs':len(packs),
                    'scope':'development diagnostic; thresholds require independent calibration','rows':rows},indent=2)+'\n')


if __name__=='__main__':main()

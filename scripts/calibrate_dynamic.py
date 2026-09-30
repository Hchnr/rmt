"""Calibrate fixed H/P/hybrid thresholds on an isolated, pinned token split."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import torch
from transformers import AutoTokenizer
from rmt.modeling_rmt import RmtForCausalLM
from rmt.training_data import load_packs
from rmt.losses import shifted_targets


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',required=True)
    ap.add_argument('--data',default='artifacts/v0.0.4_dynamic_recurr/corpus/halt_calibration.jsonl')
    ap.add_argument('--output',required=True);ap.add_argument('--packs',type=int,default=4)
    ap.add_argument('--policy',choices=['hidden','probability','hybrid'],required=True)
    ap.add_argument('--hidden-threshold',type=float,default=.4)
    ap.add_argument('--grid',type=float,nargs='+');ap.add_argument('--length',type=int,default=512)
    ap.add_argument('--full-documents',action='store_true')
    a=ap.parse_args();torch.set_num_threads(4)
    tokenizer=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
    m=RmtForCausalLM.from_pretrained(a.model,dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).cuda().eval()
    if a.full_documents:
        documents=[json.loads(x) for x in Path(a.data).read_text().splitlines()]
        packs=[{'input_ids':torch.tensor([x['input_ids']]),'labels':torch.tensor([x['labels']]),
                'attention_mask':torch.ones(1,len(x['input_ids']),dtype=torch.long),
                'segment_ids':torch.zeros(1,len(x['input_ids']),dtype=torch.long)} for x in documents]
    else:packs=load_packs(a.data,a.length,tokenizer.pad_token_id,shuffle=False)
    if a.packs:packs=packs[:a.packs]
    if not packs:raise ValueError('Empty calibration set')
    grid=[.001,.003,.01,.02,.03,.05,.075,.1,.15,.2,.25,.3,.35,.4,.5,.7,1.] if a.policy=='hidden' else [.001,.003,.01,.02,.03,.05,.07,.1,.2,.4]
    if a.grid:grid=sorted(set(a.grid))
    if not grid or any(t<=0 for t in grid):raise ValueError('Positive threshold grid required')
    rows=[];out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    def measure(tau):
        c=m.config;c.halting_policy=a.policy;c.halt_threshold=tau if a.policy=='hidden' else a.hidden_threshold
        c.halt_relative_threshold=c.halt_threshold;c.halt_probability_threshold=tau
        hist=torch.zeros(49,dtype=torch.long,device='cuda');loss=torch.zeros(2,device='cuda');caps=torch.zeros((),device='cuda')
        torch.cuda.synchronize();start=time.monotonic()
        for batch in packs:
            batch={k:v.cuda() for k,v in batch.items()}
            result=m(**batch,use_cache=False,loss_chunk_size=64)
            n=(shifted_targets(batch['labels'],batch['attention_mask'],batch['segment_ids'])!=-100).sum()
            loss+=torch.stack((result.ce_loss*n,n.float()));valid=batch['attention_mask'].bool()
            hist+=torch.bincount(result.exit_depths[valid],minlength=49);caps+=(result.exit_reasons[valid]==2).sum()
        torch.cuda.synchronize()
        row={'threshold':tau,'hidden_threshold':c.halt_threshold,'ce':(loss[0]/loss[1]).item(),
            'targets':loss[1].item(),'mean_depth':((hist*torch.arange(49,device='cuda')).sum()/hist.sum()).item(),
            'cap_fraction':(caps/hist.sum()).item(),'histogram':hist.tolist(),'seconds':time.monotonic()-start}
        rows.append(row);print(json.dumps(row),flush=True)
        return row
    def save():
        eligible=[r for r in rows if 39<=r['mean_depth']<=41]
        chosen=min(eligible,key=lambda r:r['ce']) if eligible else None
        result={'model':a.model,'data':a.data,'data_sha256':hashlib.sha256(Path(a.data).read_bytes()).hexdigest(),
            'policy':a.policy,'packs':len(packs),'pack_length':None if a.full_documents else a.length,'full_documents':a.full_documents,'selection':'lowest calibration CE among measured 39..41 mean depth; no benchmark selection',
            'rows':rows,'selected':chosen,'status':'calibrated' if chosen else 'budget_not_met'}
        out.write_text(json.dumps(result,indent=2)+'\n')
    with torch.inference_mode():
        for tau in grid:measure(tau);save()
        # A small grid refinement; no assumption of global monotonicity.
        ordered=sorted(rows,key=lambda r:abs(r['mean_depth']-40))
        closest=ordered[0]['threshold'];index=grid.index(closest)
        neighbors=grid[max(0,index-1):min(len(grid),index+2)]
        for other in neighbors:
            if other!=closest:measure((closest+other)/2);save()


if __name__=='__main__':main()

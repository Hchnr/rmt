"""Full held-out CE and token-depth audit using untruncated independent documents."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import torch
from rmt.modeling_rmt import RmtForCausalLM
from rmt.checkpoint import load_qwen_as_rmt


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);ap.add_argument('--data',required=True)
    ap.add_argument('--output',required=True);ap.add_argument('--original',action='store_true')
    ap.add_argument('--policy',choices=['fixed','hidden','probability','hybrid']);ap.add_argument('--depth',type=int)
    ap.add_argument('--threshold',type=float);ap.add_argument('--hidden-threshold',type=float)
    ap.add_argument('--limit',type=int,default=0)
    a=ap.parse_args();torch.set_num_threads(4)
    if a.original:m,_=load_qwen_as_rmt(a.model,attn_implementation='sdpa')
    else:m=RmtForCausalLM.from_pretrained(a.model,dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True)
    m=m.cuda().eval()
    if a.policy:m.config.halting_policy=a.policy
    if a.threshold is not None:
        m.config.halt_probability_threshold=a.threshold
        if m.config.halting_policy=='hidden':m.config.halt_threshold=m.config.halt_relative_threshold=a.threshold
    if a.hidden_threshold is not None:m.config.halt_threshold=m.config.halt_relative_threshold=a.hidden_threshold
    rows=[json.loads(x) for x in Path(a.data).read_text().split('\n') if x]
    if a.limit:rows=rows[:a.limit]
    sums=torch.zeros(2,device='cuda');hist=torch.zeros(m.config.max_recurrences+1,dtype=torch.long,device='cuda')
    targets_hist=torch.zeros_like(hist);domain_sums={};details=[];started=time.monotonic()
    with torch.inference_mode():
        for i,row in enumerate(rows):
            ids=torch.tensor([row['input_ids']],device='cuda');labels=torch.tensor([row['labels']],device='cuda')
            out=m(ids,labels=labels,use_cache=False,loss_chunk_size=64,recurrence_limit=a.depth)
            n=(labels[:,1:]!=-100).sum();sums+=torch.stack((out.ce_loss*n,n.float()))
            hist+=torch.bincount(out.exit_depths.flatten(),minlength=hist.numel())
            targets_hist+=torch.bincount(out.exit_depths[:,:-1][labels[:,1:]!=-100],minlength=hist.numel())
            record={'id':row['id'],'domain':row.get('domain','unknown'),'targets':n.item(),'ce':out.ce_loss.item(),
                    'mean_depth':out.exit_depths.float().mean().item(),'cap_fraction':(out.exit_reasons==2).float().mean().item()}
            details.append(record)
            d=domain_sums.setdefault(record['domain'],[0.,0]);d[0]+=record['ce']*record['targets'];d[1]+=record['targets']
            if (i+1)%50==0:print(json.dumps({'documents':i+1,'total':len(rows),'seconds':time.monotonic()-started}),flush=True)
    if sums[1]==0:raise ValueError('No held-out targets')
    result={'status':'completed','model':a.model,'config':m.config.to_dict(),'data':a.data,
        'data_sha256':hashlib.sha256(Path(a.data).read_bytes()).hexdigest(),'documents':len(rows),
        'ce':(sums[0]/sums[1]).item(),'targets':sums[1].item(),'depth_histogram':hist.tolist(),
        'mean_depth':((hist*torch.arange(hist.numel(),device='cuda')).sum()/hist.sum()).item(),
        'supervised_depth_histogram':targets_hist.tolist(),'domain_ce':{k:v[0]/v[1] for k,v in domain_sums.items() if v[1]},
        'seconds':time.monotonic()-started,'documents_detail':details,
        'scope':'Full documents, assistant CE, independent split; no generation benchmark or broad capability superiority claim'}
    Path(a.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ['documents','ce','mean_depth','seconds']}),flush=True)


if __name__=='__main__':main()

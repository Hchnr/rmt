"""Paired pre-training depth/path diagnostic on a fixed development subset."""
import json
from pathlib import Path
import torch
from rmt.checkpoint import load_qwen_as_rmt
from rmt.runtime import enforce_gpu_scope


def main():
    enforce_gpu_scope();torch.set_num_threads(4)
    model,_=load_qwen_as_rmt('/share/project/eai_pwm/models/Qwen/Qwen3-0.6B',
                            max_recurrences=36,attn_implementation='sdpa')
    model=model.cuda().eval()
    data=Path('artifacts/v0.0.5/corpus/dev.jsonl')
    rows=[json.loads(x) for x in data.read_text().split('\n') if x][:64]
    results=[]
    with torch.inference_mode():
        for depth,path in [(28,'cycle'),(32,'tail'),(32,'cycle'),(36,'tail')]:
            model.config.recurrence_schedule=path
            details=[]
            for row in rows:
                ids=torch.tensor([row['input_ids']],device='cuda')
                labels=torch.tensor([row['labels']],device='cuda')
                out=model(ids,labels=labels,use_cache=False,loss_chunk_size=64,recurrence_limit=depth)
                details.append({'id':row['id'],'ce':out.ce_loss.item(),
                                'targets':(labels[:,1:]!=-100).sum().item()})
            ce=sum(x['ce']*x['targets'] for x in details)/sum(x['targets'] for x in details)
            results.append({'depth':depth,'path':path,'ce':ce,'documents':details})
            print(json.dumps({'depth':depth,'path':path,'ce':ce}),flush=True)
    report={'rows':results,'selection':min(results[1:3],key=lambda x:x['ce'])['path'],
            'scope':'Untrained extra-depth diagnostic on first 64 frozen dev records; not trained quality or official benchmark evidence.'}
    Path('reports/v0.0.5/path_probe.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()

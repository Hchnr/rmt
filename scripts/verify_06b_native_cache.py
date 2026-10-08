"""Compare native HF DynamicCache and capacity cache on identical decode inputs."""
import json
from pathlib import Path
import torch
from transformers.cache_utils import DynamicCache
from rmt.inference.runner import Runner
from rmt.inference.cache import NativeCapacityCache


def main():
    runner=Runner('/share/project/eai_pwm/models/Qwen/Qwen3-0.6B',backend='qwen',
                  compiled=False,attention='sdpa',deterministic=True,max_context=4096)
    rows=[]
    with torch.inference_mode():
        for size in (1,4,8):
            prompts=[runner.render([{'role':'user','content':'Explain the purpose of a KV cache. '*32}]) for _ in range(size)]
            ids=runner.tokenizer(prompts,return_tensors='pt',padding=True).input_ids.cuda()
            a=DynamicCache(config=runner.model.config);b=NativeCapacityCache(ids.shape[1]+32)
            for step in range(32):
                mask=torch.ones(size,a.get_seq_length()+ids.shape[1],device='cuda',dtype=torch.long)
                x=runner.model(ids,attention_mask=mask,past_key_values=a,use_cache=True,logits_to_keep=1).logits
                y=runner.model(ids,attention_mask=mask,past_key_values=b,use_cache=True,logits_to_keep=1).logits
                row={'batch_size':size,'step':step,'equal':torch.equal(x,y),'max_abs':(x-y).abs().max().item()}
                rows.append(row);assert row['equal'],row
                ids=x[:,-1].argmax(-1,keepdim=True)
            del a,b
    report={'status':'passed','rows':rows,'scope':'Native Qwen3-0.6B BF16 deterministic SDPA; tested batches and long prefixes.'}
    Path('reports/v0.0.5/native_cache_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':'passed','comparisons':len(rows)}),flush=True)


if __name__=='__main__':main()

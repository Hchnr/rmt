"""Real-weight reference, mixed routing, padding, cache and generation regression."""
import argparse
import json
import torch
from .runner import Runner,Generation
from ..cache import RmtCapacityCache
from ..runtime import write_report,versions


def main():
    p=argparse.ArgumentParser();p.add_argument('--rmt',default='artifacts/bootstrap/rmt-bound-4b')
    p.add_argument('--qwen',default='/share/project/eai_pwm/models/Qwen/Qwen3-4B')
    p.add_argument('--output',default='reports/v0.0.3/validation.json');a=p.parse_args()
    r=Runner(a.rmt,compiled=False);q=Runner(a.qwen,backend='qwen',compiled=False)
    prompts=[r.render([{'role':'user','content':s}]) for s in ['What is 17 plus 25?','请用一句话解释为什么天空通常是蓝色。']]
    enc=r.tokenizer(prompts,return_tensors='pt',padding=True).to(r.device)
    other=q.tokenizer(prompts,return_tensors='pt',padding=True).to(q.device)
    assert torch.equal(enc.input_ids,other.input_ids)
    ids=enc.input_ids;mask=enc.attention_mask;rc=RmtCapacityCache(ids.shape[1]+8);qc=None;rows=[]
    with torch.inference_mode():
        for step in range(5):
            pos=(mask.cumsum(-1)-1).clamp_min(0)[:,-ids.shape[1]:]
            ro=r.model(ids,attention_mask=mask,position_ids=pos,use_cache=True,past_key_values=rc)
            qo=q.model(ids,attention_mask=mask,position_ids=pos,use_cache=True,past_key_values=qc)
            equal=torch.equal(ro.logits.contiguous().view(torch.uint8),qo.logits.contiguous().view(torch.uint8))
            rows.append({'step':step,'bitwise_equal':equal});assert equal
            rc,qc=ro.past_key_values,qo.past_key_values
            ids=qo.logits[:,-1].argmax(-1)[:,None];mask=torch.cat((mask,torch.ones_like(ids)),1)
    cfg=Generation(max_new_tokens=16,temperature=0,presence_penalty=0)
    cached=r.generate(prompts,[cfg]*2);uncached=r.generate(prompts,[cfg]*2,use_cache=False)
    assert [x['token_ids'] for x in cached]==[x['token_ids'] for x in uncached]
    singles=[r.generate([s],[cfg])[0] for s in prompts]
    # BF16 batch GEMM selection may change rounding; record rather than assert a
    # universal bitwise guarantee across different batch shapes.
    batch_equal=[x['token_ids']==y['token_ids'] for x,y in zip(cached,singles)]
    r.model.model.cell.bank.compile_projections()
    fast=r.generate(prompts,[cfg]*2)
    assert [x['token_ids'] for x in cached]==[x['token_ids'] for x in fast]
    # Mixed forced routes exercise every expert while holding dispatch constant.
    torch.manual_seed(171)
    ids=torch.randint(10,100000,(2,32),device=r.device)
    routes=torch.randint(0,r.model.config.num_experts,(2,32,r.model.config.num_recurrences),device=r.device)
    bank=r.model.model.cell.bank
    with torch.inference_mode():
        bank.compiled=False
        eager=r.model(ids,routing_mode='forced',forced_routes=routes,use_cache=False).logits
        bank.compiled=True
        compiled=r.model(ids,routing_mode='forced',forced_routes=routes,use_cache=False).logits
        error=(eager.float()-compiled.float());rmse=float(error.square().mean().sqrt()/eager.float().square().mean().sqrt())
        assert rmse<=1e-3
    out={'status':'passed','environment':versions(),'prefill_and_decode':rows,'tokenizer_equal':True,
         'cached_uncached_greedy_equal':True,'compiled_cached_greedy_equal':True,'batch_single_greedy_equal':batch_equal,
         'mixed_forced':{'max_abs':float(error.abs().max()),'relative_rmse':rmse,'seed':171,'shape':[2,32]},
         'scope':'Observed BF16 eager Qwen vs RMT fixed-route input shapes only; no universal bitwise claim.'}
    write_report(a.output,out);print(json.dumps(out),flush=True)


if __name__=='__main__':main()

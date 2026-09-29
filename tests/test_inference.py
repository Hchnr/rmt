import pytest
import torch
from rmt.cache import RmtCapacityCache
from rmt.inference.runner import Generation,sample


def test_capacity_cache_mixed_route_and_reorder(pair):
    _,m=pair;m.eval();ids=torch.tensor([[5,7,10,20],[12,18,21,30]])
    routes=(torch.arange(24).reshape(2,4,3)//2)%3
    with torch.inference_mode():
        full=m(ids,forced_routes=routes,routing_mode='forced',use_cache=False).logits
        cache=RmtCapacityCache(4);outs=[]
        for start,end in [(0,2),(2,3),(3,4)]:
            o=m(ids[:,start:end],forced_routes=routes[:,start:end],routing_mode='forced',use_cache=True,past_key_values=cache)
            outs.append(o.logits)
        torch.testing.assert_close(torch.cat(outs,1),full,atol=1e-5,rtol=1e-4)
        assert len(cache.storage)==3 and all(v==4 for v in cache.lengths.values())
        cache.batch_select_indices(torch.tensor([1,0,1]))
        assert all(k.shape[0]==3 for k,v in cache.storage.values())
        with pytest.raises(ValueError,match='capacity'):
            m(ids[:,:1],use_cache=True,past_key_values=cache)
        cache.reset();assert cache.get_seq_length()==0


def test_sampling_presence_and_seed():
    cfg=Generation(temperature=0,presence_penalty=1.5)
    assert sample(torch.tensor([2.,1.,0.]),[0],cfg,None)==1
    cfg=Generation(temperature=.7,top_p=.8,top_k=2)
    a=torch.Generator().manual_seed(9);b=torch.Generator().manual_seed(9)
    assert [sample(torch.tensor([2.,1.,0.]),[],cfg,a) for _ in range(20)]==[sample(torch.tensor([2.,1.,0.]),[],cfg,b) for _ in range(20)]
    with pytest.raises(ValueError): Generation(top_p=0).validate()


def test_generation_rejects_invalid_numbers():
    for kwargs in [{'temperature':float('nan')},{'temperature':float('inf')},
                   {'max_new_tokens':1.5},{'top_k':2.5},{'max_new_tokens':True}]:
        with pytest.raises(ValueError):Generation(**kwargs).validate()


def test_capacity_reuse_rejects_broadcast():
    cache=RmtCapacityCache(4)
    x=torch.zeros(2,2,1,8)
    cache.update(x,x,0)
    cache.reset()
    with pytest.raises(ValueError,match='shape'):
        cache.update(x[:1],x[:1],0)


def test_sorted_grouping_preserves_token_order(pair):
    _,m=pair
    groups=m.model.cell.bank.groups(torch.tensor([[2,0,2],[1,0,1]]))
    assert [(e,p.tolist()) for e,p in groups]==[(0,[1,4]),(1,[3,5]),(2,[0,2])]


def test_batch_early_finish_padding_and_cache(pair,monkeypatch):
    from rmt.inference.runner import Runner
    class Tokens:
        def encode(self,text,add_special_tokens=False):return list(map(int,text.split()))
        def decode(self,ids,skip_special_tokens=True):return ' '.join(map(str,ids))
    _,model=pair
    runner=Runner.__new__(Runner)
    runner.model=model.eval();runner.device=torch.device('cpu');runner.backend='rmt'
    runner.tokenizer=Tokens();runner.pad=0;runner.eos=set();runner.max_context=32
    monkeypatch.setattr(torch.cuda,'synchronize',lambda *args:None)
    prompts=['4 5','7 8 9 10'];configs=[Generation(max_new_tokens=1,temperature=0,presence_penalty=0),Generation(max_new_tokens=4,temperature=0,presence_penalty=0)]
    cached=runner.generate(prompts,configs);full=runner.generate(prompts,configs,use_cache=False)
    alone=runner.generate(prompts[1:],[configs[1]])[0]
    assert [x['token_ids'] for x in cached]==[x['token_ids'] for x in full]
    assert cached[1]['token_ids']==alone['token_ids']
    assert [x['completion_tokens'] for x in cached]==[1,4]
    assert [x['prompt_tokens'] for x in cached]==[2,4]
    stop=str(cached[1]['token_ids'][0])
    stopped=runner.generate(prompts[1:],[Generation(max_new_tokens=4,temperature=0,presence_penalty=0,stop=(stop,))])[0]
    assert stopped['finish_reason']=='stop' and stopped['text']=='' and stopped['completion_tokens']==1


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA graph integration requires a GPU')
def test_compiled_outputs_survive_next_invocation_and_weight_update(pair):
    from rmt.experts import project_qkv,project_output
    _,model=pair;model=model.to(device='cuda',dtype=torch.bfloat16).eval()
    bank=model.model.cell.bank;bank.compile_projections()
    ids=torch.tensor([[7,9,12]],device='cuda')
    with torch.inference_mode():
        first=model(ids,use_cache=False,output_hidden_states=True)
        snapshots=[x.clone() for x in first.hidden_states]
        logits=first.logits.clone()
        model(ids+1,use_cache=False)
        assert torch.equal(first.logits,logits)
        assert all(torch.equal(x,y) for x,y in zip(first.hidden_states,snapshots))
        bank.experts[0].self_attn.o_proj.weight.zero_()
        compiled=model(ids,use_cache=False).logits.clone()
        bank.compiled=False;bank._project_qkv=project_qkv;bank._project_output=project_output
        eager=model(ids,use_cache=False).logits
        torch.testing.assert_close(compiled,eager,rtol=1e-3,atol=1e-3)

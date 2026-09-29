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

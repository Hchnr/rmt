import copy
from types import MethodType
import torch
from rmt.experts import project_qkv, project_output
from rmt.losses import distillation_loss
from rmt.data import pack_sequences


def test_mixed_dispatch_against_tokenwise_reference(pair):
    _, model = pair
    reference = copy.deepcopy(model)
    def slow_qkv(bank, hidden, groups, uniform_expert=None):
        flat=hidden.reshape(-1,hidden.shape[-1])
        choices={int(i):e for e,positions in groups for i in positions}
        outputs=[project_qkv(bank.experts[choices[i]],flat[i:i+1]) for i in range(len(flat))]
        return tuple(torch.cat([out[j] for out in outputs],0).reshape(*hidden.shape[:-1],-1) for j in range(3))
    def slow_output(bank, attended, hidden, groups, uniform_expert=None):
        flat=hidden.reshape(-1,hidden.shape[-1]); a=attended.reshape(-1,attended.shape[-1])
        choices={int(i):e for e,positions in groups for i in positions}
        return torch.cat([project_output(bank.experts[choices[i]],a[i:i+1],flat[i:i+1]) for i in range(len(flat))],0).view_as(hidden)
    reference.model.cell.bank.qkv=MethodType(slow_qkv,reference.model.cell.bank)
    reference.model.cell.bank.output=MethodType(slow_output,reference.model.cell.bank)
    ids=torch.tensor([[5,7,10,20],[6,8,11,21]])
    routes=torch.tensor([[[0,0,0],[1,2,1],[2,2,0],[0,1,2]],[[2,0,2],[1,1,1],[0,2,1],[2,1,0]]])
    a=model(ids,labels=ids,forced_routes=routes,routing_mode='forced',use_cache=False)
    b=reference(ids,labels=ids,forced_routes=routes,routing_mode='forced',use_cache=False)
    torch.testing.assert_close(a.logits,b.logits,atol=1e-6,rtol=1e-4)
    a.loss.backward(); b.loss.backward()
    for (n,p),(m,q) in zip(model.named_parameters(),reference.named_parameters()):
        assert n==m
        if p.grad is not None: torch.testing.assert_close(p.grad,q.grad,atol=1e-6,rtol=1e-4)


def test_chunked_ce_kl_value_and_gradient():
    torch.manual_seed(8)
    h=torch.randn(2,5,8,requires_grad=True);w=torch.randn(13,8,requires_grad=True)
    th=torch.randn(2,5,8);tw=torch.randn(13,8)
    labels=torch.randint(0,13,(2,5)); labels[0,3]=-100
    kwargs=dict(teacher_hidden=th,teacher_weight=tw,temperature=2.0,ce_weight=0.7,kd_weight=0.3)
    a,_,_=distillation_loss(h,w,labels,chunk_size=2,**kwargs)
    ga=torch.autograd.grad(a,(h,w))
    b,_,_=distillation_loss(h,w,labels,chunk_size=100,recompute=False,**kwargs)
    gb=torch.autograd.grad(b,(h,w))
    torch.testing.assert_close(a,b)
    for x,y in zip(ga,gb): torch.testing.assert_close(x,y,atol=1e-6,rtol=1e-5)


def test_task_only_router_learning_changes_routes(pair):
    _,model=pair
    model.config.routing_mode='learned'
    model.model.cell.router.prior_strength=0.0
    ids=torch.tensor([[5,7,10,20],[4,11,21,34]])
    initial=model(ids,use_cache=False,output_router_trace=True).router_indices
    opt=torch.optim.AdamW(model.parameters(),lr=0.01)
    losses=[]; maximum_router_grad=0
    for _ in range(25):
        opt.zero_grad()
        out=model(ids,labels=ids,use_cache=False,output_router_trace=True)
        losses.append(out.loss.item())
        out.loss.backward()
        maximum_router_grad=max(maximum_router_grad,model.model.cell.router.weight.grad.norm().item())
        opt.step()
    final=model(ids,use_cache=False,output_router_trace=True).router_indices
    assert maximum_router_grad>0
    assert any(not torch.equal(x,y) for x,y in zip(initial,final))
    assert losses[-1]<losses[0]


def test_chunked_model_loss_matches_full_and_packing(pair):
    _,model=pair
    data=pack_sequences([[5,7,10],[20,30]])
    a=model(**data,use_cache=False)
    b=model(**data,use_cache=False,loss_chunk_size=2)
    torch.testing.assert_close(a.loss,b.loss)
    ga=torch.autograd.grad(a.loss,model.model.embed_tokens.weight,retain_graph=True)[0]
    gb=torch.autograd.grad(b.loss,model.model.embed_tokens.weight)[0]
    torch.testing.assert_close(ga,gb,atol=1e-6,rtol=1e-4)


def test_straight_through_proxy_gradient():
    from rmt.routing import selected_probability_st
    h=torch.tensor([[[1.,2.]]],requires_grad=True)
    z=torch.tensor([[[4.,6.]]],requires_grad=True)
    probability=torch.tensor([[.7]],requires_grad=True)
    result=selected_probability_st(z,h,probability)
    assert torch.equal(result,z)
    weights=torch.tensor([[[2.,3.]]])
    (result*weights).sum().backward()
    torch.testing.assert_close(probability.grad,torch.tensor([[18.]]))
    torch.testing.assert_close(z.grad,weights)
    torch.testing.assert_close(h.grad,torch.zeros_like(h))

import copy
import torch
from rmt.data import pack_sequences
from rmt.cache import RmtCache


def test_fixed_path_forward_backward(pair):
    teacher, model = pair
    ids = torch.tensor([[4, 12, 30, 8], [14, 2, 7, 9]])
    a = teacher(ids, labels=ids, use_cache=False, output_hidden_states=True)
    b = model(ids, labels=ids, use_cache=False, output_hidden_states=True)
    torch.testing.assert_close(a.logits, b.logits, atol=1e-5, rtol=1e-4)
    for x, y in zip(a.hidden_states, b.hidden_states):
        torch.testing.assert_close(x, y, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(a.loss, b.loss)
    a.loss.backward(); b.loss.backward()
    targets = dict(model.named_parameters())
    for name, parameter in teacher.named_parameters():
        mapped = name.replace('model.layers.', 'model.cell.bank.experts.')
        torch.testing.assert_close(parameter.grad, targets[mapped].grad, atol=1e-5, rtol=1e-4)
    assert model.lm_head.weight is model.model.embed_tokens.weight


def test_forced_fixed_path_and_repeated_experts(pair):
    _, model = pair
    ids = torch.tensor([[5, 7, 10, 20]])
    order = torch.arange(3).expand(1, 4, 3)
    a = model(ids, use_cache=False)
    b = model(ids, forced_routes=order, routing_mode='forced', use_cache=False)
    torch.testing.assert_close(a.logits, b.logits)
    routes = torch.tensor([[[0, 0, 0], [1, 0, 1], [2, 2, 1], [1, 2, 0]]])
    c = model(ids, labels=ids, forced_routes=routes, routing_mode='forced', output_router_trace=True, use_cache=False)
    c.loss.backward()
    for r, chosen in enumerate(c.router_indices):
        assert torch.equal(chosen, routes[..., r])
    for expert in model.model.cell.bank.experts:
        for p in expert.parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all()


def test_causality_and_packing(pair):
    _, model = pair
    model.eval()
    ids = torch.tensor([[5, 7, 10, 20]])
    changed = ids.clone(); changed[0, -1] = 45
    with torch.no_grad():
        for mode in ('layer_order', 'learned'):
            a = model(ids, routing_mode=mode, use_cache=False).logits
            b = model(changed, routing_mode=mode, use_cache=False).logits
            torch.testing.assert_close(a[:, :-1], b[:, :-1])
    packed = pack_sequences([[5, 7, 10], [20, 30]], length=7)
    batched = model(**packed, use_cache=False)
    first = model(torch.tensor([[5, 7, 10]]), labels=torch.tensor([[5, 7, 10]]), use_cache=False)
    second = model(torch.tensor([[20, 30]]), labels=torch.tensor([[20, 30]]), use_cache=False)
    torch.testing.assert_close(batched.logits[:, :3], first.logits)
    torch.testing.assert_close(batched.logits[:, 3:5], second.logits)
    torch.testing.assert_close(batched.loss, (2*first.loss + second.loss)/3)
    batched.loss.backward()
    grads = {n:p.grad.clone() for n,p in model.named_parameters() if p.grad is not None}
    model.zero_grad()
    ((2*first.loss+second.loss)/3).backward()
    for name,p in model.named_parameters():
        if name in grads:
            torch.testing.assert_close(p.grad, grads[name], atol=1e-6, rtol=1e-4)


def test_cache_mixed_routes_and_chunking(pair):
    _, model = pair
    model.eval()
    ids = torch.tensor([[5, 7, 10, 20, 25]])
    routes = torch.tensor([[[0,1,0], [1,0,1], [2,2,1], [1,2,0], [0,1,2]]])
    with torch.no_grad():
        full = model(ids, forced_routes=routes, routing_mode='forced', use_cache=False).logits
        cache = RmtCache(); outputs=[]
        for start, end in [(0,2),(2,4),(4,5)]:
            part = model(ids[:,start:end], forced_routes=routes[:,start:end], routing_mode='forced',
                         past_key_values=cache, use_cache=True)
            outputs.append(part.logits)
        torch.testing.assert_close(full, torch.cat(outputs,1), atol=1e-5, rtol=1e-4)
        assert len(cache.key_cache) == 3
        assert all(t.shape[-2] == 5 for t in cache.key_cache)
        assert cache.key_cache[0].data_ptr() != cache.key_cache[1].data_ptr()


def test_checkpoint_gradients_and_router(pair):
    _, model = pair
    model.config.routing_mode = 'learned'
    model.model.cell.router.prior_strength=0.0
    with torch.no_grad():
        model.model.cell.router.weight.normal_(std=0.01)
    other = copy.deepcopy(model)
    other.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    ids = torch.tensor([[5,7,10,20]])
    a=model(ids,labels=ids,use_cache=False)
    b=other(ids,labels=ids,use_cache=False)
    a.loss.backward(); b.loss.backward()
    torch.testing.assert_close(a.loss,b.loss)
    for (name,p),(other_name,q) in zip(model.named_parameters(),other.named_parameters()):
        assert name==other_name
        if p.grad is not None:
            torch.testing.assert_close(p.grad,q.grad,atol=1e-6,rtol=1e-4)
    grad=model.model.cell.router.weight.grad
    assert torch.isfinite(grad).all() and grad.abs().sum()>0


def test_batched_generate_and_hf_roundtrip(pair,tmp_path):
    from transformers import AutoModelForCausalLM
    from rmt.checkpoint import export_hf
    _,model=pair
    model.eval()
    ids=torch.tensor([[0,0,5,7],[12,20,25,30]])
    mask=ids.ne(0).long()
    with torch.no_grad():
        batch=model.generate(ids,attention_mask=mask,max_new_tokens=3,do_sample=False,eos_token_id=None)
        for i in range(2):
            single=model.generate(ids[i:i+1,mask[i].bool()],max_new_tokens=3,do_sample=False,eos_token_id=None)
            assert torch.equal(batch[i,-3:],single[0,-3:])
        right=torch.tensor([[5,7,0,0],[12,20,25,30]])
        generated=model.generate(right,attention_mask=right.ne(0).long(),max_new_tokens=3,do_sample=False,eos_token_id=None)
        assert torch.equal(batch[:,-3:],generated[:,-3:])
        export_hf(model,tmp_path)
        restored=AutoModelForCausalLM.from_pretrained(tmp_path,local_files_only=True,trust_remote_code=True)
        restored.eval()
        torch.testing.assert_close(model(ids,attention_mask=mask,use_cache=False).logits,
                                   restored(ids,attention_mask=mask,use_cache=False).logits)
        assert restored.lm_head.weight is restored.model.embed_tokens.weight

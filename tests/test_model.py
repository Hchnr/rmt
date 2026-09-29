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
        assert len(cache.layers) == 3
        assert all(t.shape[-2] == 5 for t in [layer.keys for layer in cache.layers])
        assert cache.layers[0].keys.data_ptr() != cache.layers[1].keys.data_ptr()


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


def test_cache_reorder_compact_and_sdpa(pair):
    _, model=pair
    model.eval(); model.config._attn_implementation='sdpa'
    model.config.routing_mode='learned'
    model.model.cell.router.prior_strength=0.0
    with torch.no_grad(): model.model.cell.router.weight.normal_(std=0.02)
    ids=torch.tensor([[5,7,10],[12,20,25],[4,9,30]])
    tail=torch.tensor([[13],[14]])
    order=torch.tensor([2,0])
    with torch.no_grad():
        prefill=model(ids,use_cache=True)
        cache=prefill.past_key_values
        cache.batch_select_indices(order)
        continuation=model(tail,past_key_values=cache,use_cache=True).logits
        full=model(torch.cat([ids[order],tail],dim=1),use_cache=False).logits[:,-1:]
        torch.testing.assert_close(continuation,full,atol=1e-5,rtol=1e-4)
        assert all(layer.keys.shape[0]==2 for layer in cache.layers)
        for i in range(2):
            single=model(torch.cat([ids[order[i]:order[i]+1],tail[i:i+1]],dim=1),use_cache=False).logits[:,-1:]
            torch.testing.assert_close(continuation[i:i+1],single,atol=1e-5,rtol=1e-4)


def test_eos_finishes_rows_independently(pair):
    from transformers import LogitsProcessor, LogitsProcessorList
    class DeterministicEOS(LogitsProcessor):
        def __call__(self, ids, scores):
            scores.fill_(-float('inf'))
            scores[0,2]=0
            scores[1,2 if ids.shape[1]>=4 else 11]=0
            return scores
    _,model=pair
    model.eval()
    ids=torch.tensor([[5,7],[12,20]])
    result=model.generate(ids,attention_mask=torch.ones_like(ids),max_new_tokens=5,do_sample=False,
        eos_token_id=2,pad_token_id=0,logits_processor=LogitsProcessorList([DeterministicEOS()]))
    assert result.tolist()==[[5,7,2,0,0],[12,20,11,11,2]]


def test_cache_rejects_non_append_positions(pair):
    import pytest
    _,model=pair
    model.eval()
    with pytest.raises(ValueError,match='contiguous'):
        model(torch.tensor([[5,7]]),cache_position=torch.tensor([1,2]),use_cache=True)


def test_expert_count_independent_from_recurrence_count(pair):
    from rmt.checkpoint import from_qwen_model
    teacher,_=pair
    for depth in (2,5):
        model=from_qwen_model(teacher,num_recurrences=depth).eval()
        assert len(model.model.cell.bank.experts)==3
        assert model.config.num_recurrences==depth
        assert len(model.config.layer_types)==depth
        ids=torch.tensor([[5,7,10]])
        with torch.no_grad():
            expected=model(ids,use_cache=False).logits
            prefill=model(ids[:,:2],use_cache=True)
            actual=model(ids[:,2:],past_key_values=prefill.past_key_values,use_cache=True)
        assert len(actual.past_key_values.layers)==depth
        torch.testing.assert_close(actual.logits,expected[:,2:],atol=1e-5,rtol=1e-4)

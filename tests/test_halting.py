import copy
import pytest
import torch
from rmt.cache import RmtCapacityCache
from rmt.configuration_rmt import RmtConfig
from rmt.halting import prior_expert, hidden_stable
from rmt.modeling_rmt import RmtForCausalLM
import rmt.modeling_rmt as modeling


def make_model(policy='hidden'):
    torch.manual_seed(42)
    torch.set_num_threads(2)
    config = RmtConfig(vocab_size=97, hidden_size=32, intermediate_size=48,
        num_hidden_layers=3, num_experts=3, num_recurrences=3, max_recurrences=6,
        min_recurrences=2, halt_patience=2, halting_policy=policy,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        attention_dropout=0., tie_word_embeddings=True, pad_token_id=0,
        halt_threshold=100., halt_relative_threshold=100., halt_probability_threshold=100.)
    config._attn_implementation='eager'
    return RmtForCausalLM(config).eval()


@pytest.mark.parametrize('policy', ['hidden','probability','hybrid'])
def test_dynamic_cached_chunked_and_causal(policy):
    model=make_model(policy)
    ids=torch.tensor([[4,7,9,11,13]])
    with torch.no_grad():
        full=model(ids,use_cache=False)
        cache=RmtCapacityCache(10)
        chunks=[];depths=[]
        for a,b in [(0,2),(2,3),(3,5)]:
            out=model(ids[:,a:b],past_key_values=cache,use_cache=True)
            chunks.append(out.logits);depths.append(out.exit_depths)
        torch.testing.assert_close(torch.cat(chunks,1),full.logits,atol=1e-6,rtol=1e-5)
        assert torch.equal(torch.cat(depths,1),full.exit_depths)
        assert (full.exit_depths < 6).all()
        assert all(cache.get_seq_length(i)==5 for i in range(6))
        suffix=model(torch.tensor([[4,7,21,22,23]]),use_cache=False)
        torch.testing.assert_close(full.logits[:,:2],suffix.logits[:,:2],atol=0,rtol=0)


def test_all_halted_suffix_preserves_cache_states_and_router_trace():
    model=make_model('hidden')
    ids=torch.tensor([[4,7,9]])
    dynamic_cache=RmtCapacityCache(8);fixed_cache=RmtCapacityCache(8)
    calls=[]
    handle=model.model.cell.register_forward_hook(lambda *args:calls.append(1))
    with torch.no_grad():
        dynamic=model(ids,past_key_values=dynamic_cache,use_cache=True,
                      output_hidden_states=True,output_router_trace=True)
        dynamic_calls=len(calls);calls.clear()
        fixed=model(ids,past_key_values=fixed_cache,use_cache=True,
                    recurrence_limit=2,halting_policy='fixed',
                    output_hidden_states=True,output_router_trace=True)
    handle.remove()
    assert dynamic_calls==2
    torch.testing.assert_close(dynamic.logits,fixed.logits,atol=0,rtol=0)
    assert torch.equal(dynamic.exit_depths,torch.full_like(ids,2))
    assert len(dynamic.hidden_states)==model.config.max_recurrences+1
    assert len(dynamic.router_indices)==model.config.max_recurrences
    assert len(dynamic.router_probabilities)==model.config.max_recurrences
    for a,b in zip(dynamic.hidden_states[:2],fixed.hidden_states[:2]):
        torch.testing.assert_close(a,b,atol=0,rtol=0)
    assert all(torch.equal(state,dynamic.hidden_states[2]) for state in dynamic.hidden_states[2:-1])
    for route,reference in zip(dynamic.router_indices[:2],fixed.router_indices):
        assert torch.equal(route,reference)
    assert all((route==-1).all() for route in dynamic.router_indices[2:])
    assert all(probability.numel()==0 for probability in dynamic.router_probabilities[2:])
    assert all(dynamic_cache.get_seq_length(i)==ids.shape[1] for i in range(model.config.max_recurrences))
    for layer in range(2,model.config.max_recurrences):
        for actual,reference in zip(dynamic_cache.storage[layer],dynamic_cache.storage[1]):
            assert torch.equal(actual[...,:ids.shape[1],:],reference[...,:ids.shape[1],:])
    for layer in range(2):
        for actual,reference in zip(dynamic_cache.storage[layer],fixed_cache.storage[layer]):
            assert torch.equal(actual[...,:ids.shape[1],:],reference[...,:ids.shape[1],:])


def test_all_halted_suffix_preserves_training_gradients():
    dynamic=make_model('hidden').train();fixed=copy.deepcopy(dynamic)
    ids=torch.tensor([[4,7,9,11]])
    dynamic(ids,labels=ids,use_cache=False).loss.backward()
    fixed(ids,labels=ids,use_cache=False,halting_policy='fixed',recurrence_limit=2).loss.backward()
    for a,b in zip(dynamic.parameters(),fixed.parameters()):
        torch.testing.assert_close(a.grad,b.grad,atol=0,rtol=0)


def test_heterogeneous_early_halts_fill_each_tokens_final_kv():
    model=make_model('hidden')
    ids=torch.tensor([[4,7,9,11]])
    stops=torch.tensor([[1,4,2,5]])
    calls=[]
    handle=model.model.cell.register_forward_hook(lambda *args:calls.append(1))
    cache=RmtCapacityCache(8)
    with torch.no_grad():
        full=model(ids,forced_exit_depths=stops,use_cache=False)
        cached=model(ids,forced_exit_depths=stops,past_key_values=cache,use_cache=True)
    handle.remove()
    assert len(calls)==10  # five evaluated steps in each of the two runs
    torch.testing.assert_close(full.logits,cached.logits,atol=0,rtol=0)
    assert torch.equal(cached.exit_depths,stops)
    assert all(cache.get_seq_length(i)==ids.shape[1] for i in range(model.config.max_recurrences))
    for token,depth in enumerate(stops[0].tolist()):
        for layer in range(depth,model.config.max_recurrences):
            for actual,reference in zip(cache.storage[layer],cache.storage[depth-1]):
                assert torch.equal(actual[...,token:token+1,:],reference[...,token:token+1,:])


def test_heterogeneous_stops_retain_history_and_gradients():
    model=make_model().train()
    ids=torch.tensor([[4,7,9,11]])
    stops=torch.tensor([[1,6,2,5]])
    inputs=model.get_input_embeddings()(ids).detach().requires_grad_()
    out=model(inputs_embeds=inputs,forced_exit_depths=stops,use_cache=False)
    out.logits[:,-1].square().sum().backward()
    assert inputs.grad[0,0].abs().sum()>0  # later query still reads early-exit KV
    reference=copy.deepcopy(model).eval()
    with torch.no_grad():
        cache=RmtCapacityCache(8);logits=[]
        for t in range(4):
            one=reference(ids[:,t:t+1],forced_exit_depths=stops[:,t:t+1],past_key_values=cache,use_cache=True)
            logits.append(one.logits)
        torch.testing.assert_close(torch.cat(logits,1),out.logits,atol=1e-6,rtol=1e-5)
        assert not torch.equal(cache.storage[0][0][:,:,:4],cache.storage[5][0][:,:,:4])
        torch.testing.assert_close(cache.storage[0][0][:,:,0],cache.storage[5][0][:,:,0])
    assert torch.equal(out.exit_depths,stops)


def test_checkpoint_gradients_and_forced_trajectory_match():
    a=make_model().train();b=copy.deepcopy(a)
    b.gradient_checkpointing_enable()
    ids=torch.tensor([[4,7,9,11]])
    stops=torch.tensor([[1,6,2,5]])
    for m in (a,b):m(ids,labels=ids,use_cache=False,forced_exit_depths=stops).loss.backward()
    for pa,pb in zip(a.parameters(),b.parameters()):
        torch.testing.assert_close(pa.grad,pb.grad,atol=0,rtol=0)


def test_full_depth_forcing_matches_fixed_and_no_halt_is_cap():
    model=make_model()
    ids=torch.tensor([[4,7,9]])
    with torch.no_grad():
        fixed=model(ids,halting_policy='fixed',recurrence_limit=6,use_cache=False)
        forced=model(ids,forced_exit_depths=torch.full_like(ids,6),use_cache=False)
        torch.testing.assert_close(fixed.logits,forced.logits,atol=0,rtol=0)
        model.config.halt_threshold=0
        capped=model(ids,use_cache=False)
        assert (capped.exit_depths==6).all() and (capped.exit_reasons==2).all()


def test_padding_packing_and_tail_schedule():
    model=make_model()
    ids=torch.tensor([[4,7,9,11,0]])
    seg=torch.tensor([[0,0,1,1,-1]])
    with torch.no_grad():
        packed=model(ids,attention_mask=(ids!=0).long(),segment_ids=seg,use_cache=False)
        one=model(ids[:,2:4],use_cache=False)
        torch.testing.assert_close(packed.logits[:,2:4],one.logits,atol=1e-6,rtol=1e-5)
    assert packed.exit_depths[0,-1]==0
    model.config.recurrence_schedule='tail';model.config.tail_experts=2
    assert [prior_expert(model.config,i) for i in range(6)]==[0,1,2,1,2,1]


def test_config_roundtrip_and_invalid_controls(tmp_path):
    model=make_model();model.save_pretrained(tmp_path)
    restored=RmtForCausalLM.from_pretrained(tmp_path)
    assert restored.config.max_recurrences==6
    assert restored.model.cell.router.step_bias.shape==(6,3)
    ids=torch.tensor([[4,7]])
    with torch.no_grad():
        torch.testing.assert_close(model(ids,use_cache=False).logits,restored(ids,use_cache=False).logits)
    with pytest.raises(ValueError):model(ids,recurrence_limit=7)
    with pytest.raises(ValueError):model(ids,forced_exit_depths=torch.zeros_like(ids))
    with pytest.raises(FloatingPointError):hidden_stable(torch.ones(1,32),torch.full((1,32),float('nan')),model.config)


def test_cache_refuses_missing_deeper_history_and_can_reset():
    model=make_model();ids=torch.tensor([[4,7]])
    cache=RmtCapacityCache(8)
    with torch.no_grad():
        model(ids,halting_policy='fixed',recurrence_limit=3,past_key_values=cache,use_cache=True)
        with pytest.raises(ValueError,match='Incomplete recurrence cache'):
            model(ids[:,:1],past_key_values=cache,use_cache=True)
        cache.reset()
        out=model(ids,past_key_values=cache,use_cache=True)
        assert (out.exit_depths==2).all()
        cache.batch_select_indices(torch.tensor([0,0]))
        assert cache.storage[5][0].shape[0]==2


def test_curriculum_is_rank_independent_and_replayable():
    from rmt.train import recurrence_controls
    cfg={'seed':17,'recurrence_training':{'mode':'random','depths':[36,40,44,48],
         'probabilities':[.4,.3,.2,.1],'fixed_warmup_steps':3}}
    a=[recurrence_controls(cfg,i) for i in range(100)]
    torch.manual_seed(99)
    b=[recurrence_controls(cfg,i) for i in range(100)]
    assert a==b and all(x['recurrence_limit']==36 for x in a[:3])
    assert {x['recurrence_limit'] for x in a}=={36,40,44,48}


def test_only_extra_recurrences_open_routing():
    model=make_model();model.config.learned_routing_start=3
    model.model.cell.router.prior_strength=0
    with torch.no_grad():
        model.model.cell.router.step_bias[:,2]=100
        out=model(torch.tensor([[4,7]]),halting_policy='fixed',recurrence_limit=6,
                  routing_mode='learned',output_router_trace=True,use_cache=False)
    assert [r[0,0].item() for r in out.router_indices]==[0,1,2,2,2,2]


def test_inactive_tokens_skip_all_seven_projection_calls():
    model=make_model();counts={name:0 for name in ['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']}
    handles=[]
    def count(name):
        def hook(module,args):counts[name]+=args[0].numel()//args[0].shape[-1]
        return hook
    for expert in model.model.cell.bank.experts:
        for name in counts:
            owner=expert.self_attn if name in ['q_proj','k_proj','v_proj','o_proj'] else expert.mlp
            handles.append(getattr(owner,name).register_forward_pre_hook(count(name)))
    stops=torch.tensor([[2,4,3,5]])
    with torch.no_grad():model(torch.tensor([[4,7,9,11]]),forced_exit_depths=stops,use_cache=False)
    for handle in handles:handle.remove()
    assert counts==dict.fromkeys(counts,stops.sum().item())


def test_cache_select_reorder_and_decode_preserve_heterogeneous_history():
    model=make_model();ids=torch.tensor([[4,7,9],[11,13,17]])
    stops=torch.tensor([[2,6,3],[5,1,4]]);cache=RmtCapacityCache(8)
    selection=torch.tensor([1,0,1]);next_ids=torch.tensor([[19],[23],[29]]);next_stops=torch.tensor([[6],[1],[4]])
    with torch.no_grad():
        model(ids,forced_exit_depths=stops,past_key_values=cache,use_cache=True)
        cache.reorder_cache(selection)
        actual=model(next_ids,forced_exit_depths=next_stops,past_key_values=cache,use_cache=True)
        expected=model(torch.cat((ids[selection],next_ids),1),
            forced_exit_depths=torch.cat((stops[selection],next_stops),1),use_cache=False)
    torch.testing.assert_close(actual.logits[:,-1],expected.logits[:,-1],atol=1e-6,rtol=1e-5)
    assert torch.equal(actual.exit_depths,next_stops)


@pytest.mark.parametrize('policy',['probability','hybrid'])
def test_deferred_probability_checks_preserve_logits_depths_and_gradients(policy,monkeypatch):
    model=make_model(policy).train();model.config.min_recurrences=5
    ids=torch.tensor([[4,7,9,11]]);labels=ids.clone()
    original=torch.nn.functional.linear;calls=[]
    def linear(x,weight,bias=None):
        if weight is model.lm_head.weight:calls.append(x.numel()//x.shape[-1])
        return original(x,weight,bias)
    monkeypatch.setattr(torch.nn.functional,'linear',linear)
    outputs=[];gradients=[];counts=[]
    for deferred in [False,True]:
        model.config.defer_probability_checks=deferred;model.zero_grad(set_to_none=True);calls.clear()
        out=model(ids,labels=labels,use_cache=False);out.loss.backward()
        outputs.append((out.logits.detach().clone(),out.exit_depths.clone(),out.exit_reasons.clone()))
        gradients.append({k:p.grad.clone() for k,p in model.named_parameters() if p.grad is not None})
        counts.append(sum(calls))
    assert counts[1]<counts[0]
    for a,b in zip(outputs[0],outputs[1]):torch.testing.assert_close(a,b,atol=0,rtol=0)
    assert gradients[0].keys()==gradients[1].keys()
    for name in gradients[0]:torch.testing.assert_close(gradients[0][name],gradients[1][name],atol=0,rtol=0)


@pytest.mark.parametrize('policy',['probability','hybrid'])
def test_probability_function_is_not_called_before_legal_halt_window(policy,monkeypatch):
    model=make_model(policy);model.config.min_recurrences=5
    original=modeling.probability_stable;calls=[]
    def counted(*args,**kwargs):
        calls.append(args[2].clone())
        return original(*args,**kwargs)
    monkeypatch.setattr(modeling,'probability_stable',counted)
    with torch.no_grad():out=model(torch.tensor([[4,7,9]]),use_cache=False)
    assert len(calls)==2
    assert all(mask.all() for mask in calls)
    assert torch.equal(out.exit_depths,torch.full((1,3),5,dtype=torch.long))

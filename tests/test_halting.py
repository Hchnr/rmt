import copy
import pytest
import torch
from rmt.cache import RmtCapacityCache
from rmt.configuration_rmt import RmtConfig
from rmt.halting import prior_expert, hidden_stable
from rmt.modeling_rmt import RmtForCausalLM


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

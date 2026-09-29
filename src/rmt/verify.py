"""Full local-weight migration and numerical equivalence acceptance entry point."""
import argparse
import gc
import json
from pathlib import Path
import time

import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer
from .checkpoint import load_qwen_as_rmt, map_key, export_hf
from .data import FIXTURES
from .runtime import enforce_gpu_scope, write_report, versions


def errors(a, b):
    a, b = a.float(), b.float()
    return {'max_abs': (a-b).abs().max().item(),
            'relative_rmse': ((a-b).square().mean().sqrt()/a.square().mean().sqrt().clamp_min(1e-12)).item()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-model', default='/share/project/eai_pwm/models/Qwen/Qwen3-4B')
    parser.add_argument('--output', default='reports/bootstrap/equivalence_4b.json')
    parser.add_argument('--export', default='artifacts/bootstrap/rmt-bound-4b')
    parser.add_argument('--reload', action='store_true')
    args = parser.parse_args()
    enforce_gpu_scope()
    torch.set_num_threads(8)
    torch.manual_seed(17)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, local_files_only=True)
    tokenizer.padding_side = 'left'
    batch = tokenizer(FIXTURES, return_tensors='pt', padding=True, truncation=True, max_length=32).to('cuda:0')
    positions = (batch.attention_mask.long().cumsum(-1)-1).clamp_min(0)
    if args.reload:
        # Deliberately run in a separate process with AutoClass remote-code loading.
        model = AutoModelForCausalLM.from_pretrained(args.export, local_files_only=True,
            trust_remote_code=True, torch_dtype=torch.bfloat16, attn_implementation='eager').to('cuda:0').eval()
        reference = torch.load(Path(args.export)/'acceptance.pt', weights_only=True)
        with torch.no_grad():
            output = model(**batch, use_cache=False).logits.cpu()
            generated = model.generate(**batch, max_new_tokens=3, do_sample=False).cpu()
        torch.testing.assert_close(output, reference['logits'], atol=0, rtol=0)
        assert torch.equal(generated, reference['generated'])
        assert model.lm_head.weight is model.model.embed_tokens.weight
        write_report(args.output, {'status':'passed','separate_process_reload':True,'environment':versions()})
        return
    started = time.monotonic()
    source = AutoModelForCausalLM.from_pretrained(args.base_model, local_files_only=True,
        torch_dtype=torch.bfloat16, attn_implementation='eager').to('cuda:0').eval()
    with torch.no_grad():
        expected = source(**batch, position_ids=positions, labels=batch.input_ids.masked_fill(batch.attention_mask==0,-100),
                          output_hidden_states=True, use_cache=False)
        expected_logits = expected.logits.cpu()
        expected_hidden = [x.cpu() for x in expected.hidden_states]
        # Our loss masks the padding->first-token boundary explicitly.
        targets = batch.input_ids[:,1:].clone()
        targets.masked_fill_(~(batch.attention_mask[:,1:].bool() & batch.attention_mask[:,:-1].bool()), -100)
        expected_loss = torch.nn.functional.cross_entropy(expected.logits[:,:-1].float().reshape(-1,source.config.vocab_size),targets.reshape(-1)).item()
    del source, expected
    gc.collect(); torch.cuda.empty_cache()
    model, mapping = load_qwen_as_rmt(args.base_model)
    params = dict(model.named_parameters(remove_duplicate=False))
    index = json.loads(Path(args.base_model,'model.safetensors.index.json').read_text())
    checked = 0
    for shard_name in sorted(set(index['weight_map'].values())):
        with safe_open(Path(args.base_model,shard_name), framework='pt') as shard:
            for name in shard.keys():
                assert torch.equal(shard.get_tensor(name),params[map_key(name)])
                checked += 1
    del params
    model = model.to('cuda:0').eval()
    with torch.no_grad():
        actual = model(**batch, labels=batch.input_ids, output_hidden_states=True, use_cache=False)
        valid = batch.attention_mask.bool().cpu()
        metric = errors(expected_logits[valid], actual.logits.cpu()[valid])
        hidden_metrics = [errors(a[valid],b.cpu()[valid]) for a,b in zip(expected_hidden,actual.hidden_states)]
        delta_nll = abs(expected_loss-actual.loss.item())
        assert metric['relative_rmse'] <= 1e-3 and delta_nll <= 1e-3, (metric,delta_nll)
        generated = model.generate(**batch,max_new_tokens=3,do_sample=False)
        for row in range(batch.input_ids.shape[0]):
            ids = batch.input_ids[row:row+1,batch.attention_mask[row].bool()]
            single = model.generate(ids, attention_mask=torch.ones_like(ids),max_new_tokens=3,do_sample=False)
            assert torch.equal(generated[row,-3:], single[0,-3:])
        acceptance = {'logits':actual.logits.cpu(),'generated':generated.cpu()}
    del actual
    model.cpu()
    export_hf(model,args.export,args.base_model)
    Path(args.export,'conversion.json').write_text(json.dumps(mapping,indent=2)+'\n')
    torch.save(acceptance,Path(args.export,'acceptance.pt'))
    report = {'status':'passed','environment':versions(),'source':args.base_model,
        'tensor_equality_checked':checked,'base_parameters':mapping['base_parameters'],
        'router_parameters':mapping['router_parameters'],'logits':metric,'delta_nll':delta_nll,
        'hidden_per_depth':hidden_metrics,'batched_generation_equal':True,
        'elapsed_seconds':time.monotonic()-started,'peak_memory_bytes':torch.cuda.max_memory_allocated(),
        'checkpoint':args.export}
    write_report(args.output,report)
    print(json.dumps(report))


if __name__ == '__main__': main()

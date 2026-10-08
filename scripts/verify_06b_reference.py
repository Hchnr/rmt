"""Native/RMT exact reference and cached decode for the pinned 0.6B model."""
import argparse
import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from rmt.checkpoint import load_qwen_as_rmt, export_hf
from rmt.runtime import enforce_gpu_scope, versions


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base', default='/share/project/eai_pwm/models/Qwen/Qwen3-0.6B')
    p.add_argument('--report', default='reports/v0.0.5/reference.json')
    a = p.parse_args()
    enforce_gpu_scope()
    torch.set_num_threads(4)
    tok = AutoTokenizer.from_pretrained(a.base, local_files_only=True)
    native = AutoModelForCausalLM.from_pretrained(a.base, local_files_only=True,
        dtype=torch.bfloat16, attn_implementation='eager').cuda().eval()
    rmt, mapping = load_qwen_as_rmt(a.base, max_recurrences=36, attn_implementation='eager')
    rmt = rmt.cuda().eval()
    checks = []
    prompts = ['Explain why the sky is blue.', '请用一句话解释循环神经网络。']
    with torch.inference_mode():
        for side in ('left', 'right'):
            tok.padding_side = side
            batch = tok(prompts, padding=True, return_tensors='pt').to('cuda')
            x = native(**batch, use_cache=False).logits
            y = rmt(**batch, use_cache=False).logits
            valid = batch.attention_mask.bool()
            equal = torch.equal(x[valid], y[valid])
            checks.append({'case':side+'_padding', 'equal':equal,
                           'max_abs':(x[valid]-y[valid]).abs().max().item()})
            assert equal, checks[-1]
        ids = tok(prompts[0], return_tensors='pt').input_ids.cuda()
        nc = rc = None
        for step in range(5):
            x = native(ids, past_key_values=nc, use_cache=True)
            y = rmt(ids, past_key_values=rc, use_cache=True)
            equal = torch.equal(x.logits, y.logits)
            checks.append({'case':f'cached_step_{step}', 'equal':equal,
                           'max_abs':(x.logits-y.logits).abs().max().item()})
            assert equal, checks[-1]
            nc, rc = x.past_key_values, y.past_key_values
            ids = x.logits[:, -1].argmax(-1, keepdim=True)
        assert rmt.lm_head.weight is rmt.model.embed_tokens.weight
        root = Path('artifacts/v0.0.5/reference_hf')
        export_hf(rmt, root, a.base, native.generation_config)
        restored = AutoModelForCausalLM.from_pretrained(root, trust_remote_code=True,
            local_files_only=True, dtype=torch.bfloat16, attn_implementation='eager').cuda().eval()
        assert restored.lm_head.weight is restored.model.embed_tokens.weight
        assert torch.equal(restored(ids, use_cache=False).logits, rmt(ids, use_cache=False).logits)
    report = {'status':'passed', 'environment':versions(), 'checks':checks,
              'hf_reload_equal':True, 'tied_weights':True,
              'base_parameters':mapping['base_parameters'], 'router_parameters':mapping['router_parameters'],
              'scope':'BF16 eager, tested inputs and batch shapes only; not cross-backend equivalence'}
    Path(a.report).write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()

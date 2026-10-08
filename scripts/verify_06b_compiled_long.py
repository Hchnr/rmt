"""Compare aligned long-prefix logits, separately from autoregressive divergence."""
import json
from pathlib import Path
import torch
from rmt.inference.runner import Runner


def main():
    runner=Runner('artifacts/v0.0.5/reference_hf',compiled=False,attention='sdpa',max_context=4096)
    runner.model.eval();references={};batches={}
    with torch.inference_mode():
        for size in (1,4,8):
            prompts=[runner.render([{'role':'user','content':' '.join([
                f'Explain in detail how to test a sorting algorithm. Include example {i+1}.']*32)}]) for i in range(size)]
            batches[size]=runner.tokenizer(prompts,padding=True,return_tensors='pt').to('cuda')
            batch=batches[size]
            batch['position_ids']=(batch.attention_mask.cumsum(-1)-1).clamp_min(0)
            references[size]=runner.model(**batch,use_cache=False,logits_to_keep=16).logits.float().cpu()
        runner.model.model.cell.bank.compile_projections()
        rows=[]
        for size,batch in batches.items():
            actual=runner.model(**batch,use_cache=False,logits_to_keep=16).logits.float().cpu()
            expected=references[size];delta=actual-expected
            row={'batch_size':size,'prompt_tokens':batch.input_ids.shape[1],
                 'max_abs':delta.abs().max().item(),
                 'relative_rmse':(delta.square().mean().sqrt()/expected.square().mean().sqrt()).item(),
                 'argmax_disagreement_fraction':(actual.argmax(-1)!=expected.argmax(-1)).float().mean().item()}
            rows.append(row)
            assert row['max_abs']<=.5 and row['relative_rmse']<.01,row
    report={'status':'passed','rows':rows,'scope':'Aligned last 16 positions of repeated long prompts; numerical tolerance, not bitwise or generated-answer equivalence.'}
    Path('reports/v0.0.5/compiled_long_numeric.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()

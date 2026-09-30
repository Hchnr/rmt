"""Resumable, sharded non-thinking sequence distillation with exact provenance."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',default='artifacts/v0.0.4_dynamic_recurr/teacher/Qwen3-32B')
    ap.add_argument('--data',default='artifacts/v0.0.4_dynamic_recurr/corpus/train.jsonl')
    ap.add_argument('--output',default='artifacts/v0.0.4_dynamic_recurr/teacher_answers')
    ap.add_argument('--rank',type=int,default=0);ap.add_argument('--world-size',type=int,default=1)
    ap.add_argument('--limit',type=int,default=1000);ap.add_argument('--batch-size',type=int,default=4)
    ap.add_argument('--max-new-tokens',type=int,default=1024)
    a=ap.parse_args();torch.set_num_threads(4)
    model_path=Path(a.model)
    if 'Qwen3-32B' in a.model and not (model_path/'download_manifest.json').exists():
        raise ValueError('Teacher download must be completely SHA256 verified before loading')
    tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True);tok.padding_side='left'
    rows=[json.loads(x) for x in Path(a.data).read_text().split('\n') if x]
    quotas=dict(zip(['general','math','code','chinese'],[int(a.limit*x) for x in [.4,.3,.2,.1]]))
    selected=[]
    for row in rows:
        if quotas.get(row['domain'],0)<=0:continue
        prompt=tok.apply_chat_template(row['messages'][:1],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        if len(tok.encode(prompt,add_special_tokens=False))>1536:continue
        row['rendered_prompt']=prompt;selected.append(row);quotas[row['domain']]-=1
    selected.sort(key=lambda x:x['id']);assigned=selected[a.rank::a.world_size]
    root=Path(a.output);root.mkdir(parents=True,exist_ok=True)
    config={'max_new_tokens':a.max_new_tokens,'do_sample':True,'temperature':.7,'top_p':.8,'top_k':20,
            'enable_thinking':False,'batch_size':a.batch_size,'seed_base':17000,'sampling_seed_scope':'batch'}
    identity={'model':a.model,'model_config_sha256':hashlib.sha256((model_path/'config.json').read_bytes()).hexdigest(),
        'teacher_manifest_sha256':hashlib.sha256((model_path/'download_manifest.json').read_bytes()).hexdigest() if (model_path/'download_manifest.json').exists() else None,
        'data_sha256':hashlib.sha256(Path(a.data).read_bytes()).hexdigest(),'generation':config,
        'selection_ids':[x['id'] for x in selected],'unfilled_quotas':quotas,'rank':a.rank,'world_size':a.world_size,
        'torch':torch.__version__,'template_sha256':hashlib.sha256(tok.chat_template.encode()).hexdigest()}
    identity_path=root/f'rank_{a.rank}_identity.json'
    if identity_path.exists() and json.loads(identity_path.read_text())!=identity:raise ValueError('Teacher generation resume identity changed')
    identity_path.write_text(json.dumps(identity,indent=2)+'\n')
    output=root/f'rank_{a.rank}.jsonl';completed={}
    if output.exists():
        for line in output.read_text().split('\n'):
            if line:result=json.loads(line);completed[result['id']]=result
    model=None;started=time.monotonic();generated=0
    eos=None
    for offset in range(0,len(assigned),a.batch_size):
        batch=assigned[offset:offset+a.batch_size]
        if all(x['id'] in completed for x in batch):continue
        if any(x['id'] in completed for x in batch):raise ValueError('Partial batch on resume; explicit recovery required to preserve batch identity')
        if model is None:
            model=AutoModelForCausalLM.from_pretrained(a.model,local_files_only=True,dtype=torch.bfloat16,
                attn_implementation='sdpa',device_map={'':'cuda:0'}).eval()
            eos=model.generation_config.eos_token_id;eos=set(eos if isinstance(eos,list) else [eos])
        encoded=tok([r['rendered_prompt'] for r in batch],add_special_tokens=False,padding=True,return_tensors='pt').to('cuda')
        seed=17000+a.rank*100000+offset;torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        t=time.monotonic()
        with torch.inference_mode():
            result=model.generate(**encoded,max_new_tokens=a.max_new_tokens,do_sample=True,temperature=.7,
                top_p=.8,top_k=20,pad_token_id=tok.pad_token_id)
        torch.cuda.synchronize();seconds=time.monotonic()-t
        records=[]
        for row,seq in zip(batch,result[:,encoded.input_ids.shape[1]:].tolist()):
            end=next((i for i,x in enumerate(seq) if x in eos),None)
            ids=seq if end is None else seq[:end]
            records.append({'id':row['id'],'domain':row['domain'],'messages':row['messages'][:1],
                'text':tok.decode(ids,skip_special_tokens=True),'token_ids':ids,'finish_reason':'length' if end is None else 'stop',
                'batch_seed':seed,'batch_ids':[r['id'] for r in batch],'batch_seconds':seconds,
                'reference_answer':row.get('reference_answer'),'source':row['source'],'source_row':row['source_row']})
        with output.open('a') as f:
            for record in records:f.write(json.dumps(record,ensure_ascii=False)+'\n')
        generated+=len(records);print(json.dumps({'rank':a.rank,'generated':generated,'assigned':len(assigned),'seconds':seconds}),flush=True)
    summary={'status':'completed','assigned':len(assigned),'new_records':generated,'wall_seconds':time.monotonic()-started}
    (root/f'rank_{a.rank}_summary.json').write_text(json.dumps(summary,indent=2)+'\n')


if __name__=='__main__':main()

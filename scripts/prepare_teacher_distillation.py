"""Audit completed teacher answers and tokenize only eligible training records."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
from transformers import AutoTokenizer
from rmt.training_data import tokenize_response


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--answers',default='artifacts/v0.0.4_dynamic_recurr/teacher_answers')
    ap.add_argument('--output',default='artifacts/v0.0.4_dynamic_recurr/corpus/teacher32_train.jsonl')
    ap.add_argument('--report',default='reports/v0.0.4_dynamic_recurr/teacher_distillation.json')
    ap.add_argument('--source',default='artifacts/v0.0.4_dynamic_recurr/corpus/train.jsonl')
    a=ap.parse_args();root=Path(a.answers)
    train_path=Path(a.source)
    train={r['id']:r for r in map(json.loads,train_path.read_text().splitlines())}
    identities=[json.loads(p.read_text()) for p in sorted(root.glob('rank_*_identity.json'))]
    if not identities:raise ValueError('No teacher identities')
    worlds={x['world_size'] for x in identities}
    if len(worlds)!=1 or {x['rank'] for x in identities}!=set(range(next(iter(worlds)))):raise ValueError('Incomplete teacher workers')
    if hashlib.sha256(train_path.read_bytes()).hexdigest()!=identities[0]['data_sha256']:raise ValueError('Training source changed since teacher generation')
    common={k:v for k,v in identities[0].items() if k!='rank'}
    if any({k:v for k,v in x.items() if k!='rank'}!=common for x in identities):raise ValueError('Teacher worker configuration mismatch')
    expected=identities[0]['selection_ids']
    if any(x['selection_ids']!=expected or x['data_sha256']!=identities[0]['data_sha256'] for x in identities):raise ValueError('Teacher selection mismatch')
    raw=[]
    for identity in identities:
        summary=json.loads((root/f'rank_{identity["rank"]}_summary.json').read_text())
        if summary['status']!='completed':raise ValueError('Incomplete teacher generation')
        part=root/f'rank_{identity["rank"]}.jsonl'
        if hashlib.sha256(part.read_bytes()).hexdigest()!=summary['output_sha256']:raise ValueError('Teacher output hash mismatch')
        shard=[json.loads(x) for x in part.read_text().splitlines()]
        if {x['id'] for x in shard}!=set(expected[identity['rank']::identity['world_size']]):raise ValueError('Worker assignment mismatch')
        raw += shard
    if len({x['id'] for x in raw})!=len(raw) or {x['id'] for x in raw}!=set(expected):raise ValueError('Missing or duplicate teacher answers')
    tokenizer=AutoTokenizer.from_pretrained('/share/project/eai_pwm/models/Qwen/Qwen3-4B',local_files_only=True)
    from math_verify import parse, verify, LatexExtractionConfig
    accepted=[];audit=[];excluded={}
    for row in raw:
        if row['id'] not in train:raise ValueError('Non-training prompt entered distillation')
        if row['messages']!=train[row['id']]['messages'][:1]:raise ValueError('Teacher prompt differs from frozen source')
        text=row['text'];reason=None;verification='format_only'
        if row['finish_reason']!='stop':reason='truncated'
        elif len(text.strip())<8:reason='empty_or_short'
        elif '<think>' in text or '</think>' in text:reason='unexpected_thinking'
        elif row['domain']=='math':
            gold=row.get('reference_answer')
            if gold is None:reason='math_missing_reference'
            else:
                try:
                    # Bare expressions can otherwise be reduced to an unrelated interior number.
                    source=str(gold).strip()
                    if '$' not in source and not source.startswith(('\\(', '\\[')):source='$'+source+'$'
                    parsed_gold=parse(source,extraction_config=[LatexExtractionConfig()])
                    matched=bool(parsed_gold) and verify(parsed_gold,parse(text))
                except Exception:matched=False
                if not matched:reason='math_unverified'
                else:verification='math_verify_against_source_answer'
        elif row['domain']=='code':
            blocks=re.findall(r'```python\s*\n(.*?)```',text,re.S)
            if blocks:
                try:
                    for block in blocks:ast.parse(block)
                    verification='python_syntax_only_no_correctness_claim'
                except SyntaxError:reason='python_syntax_error'
        messages=row['messages']+[{'role':'assistant','content':text}]
        ids,labels=tokenize_response(tokenizer,messages)
        if len(ids)>4096:reason=reason or 'student_length'
        if reason:excluded[reason]=excluded.get(reason,0)+1
        else:accepted.append({'id':row['id'],'domain':row['domain'],'input_ids':ids,'labels':labels,
            'messages':messages,'teacher_identity':identities[0]['model'],'verification':verification})
        audit.append({'id':row['id'],'domain':row['domain'],'accepted':reason is None,'reason':reason,'verification':verification})
    accepted.sort(key=lambda x:x['id']);path=Path(a.output)
    path.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in accepted))
    matched=path.with_name('matched_source_train.jsonl')
    matched.write_text(''.join(json.dumps(train[x['id']],ensure_ascii=False)+'\n' for x in accepted))
    report={'matched_source':{'path':str(matched),'sha256':hashlib.sha256(matched.read_bytes()).hexdigest(),
        'input_tokens':sum(len(train[x['id']]['input_ids']) for x in accepted),
        'targets':sum(sum(y!=-100 for y in train[x['id']]['labels']) for x in accepted)},'status':'prepared','teacher_identities':identities,'generated':len(raw),'accepted':len(accepted),
        'excluded':excluded,'input_tokens':sum(len(x['input_ids']) for x in accepted),
        'targets':sum(sum(y!=-100 for y in x['labels']) for x in accepted),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'domain_counts':{domain:{'generated':sum(x['domain']==domain for x in raw),
            'accepted':sum(x['domain']==domain for x in accepted)} for domain in sorted({x['domain'] for x in raw})},
        'audit':audit,'scope':'Math source-answer verification; code syntax and other format checks do not establish semantic correctness'}
    Path(a.report).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['generated','accepted','excluded','input_tokens','targets']}),flush=True)


if __name__=='__main__':main()

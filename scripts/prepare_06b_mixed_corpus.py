"""Broaden prompt coverage while preserving accepted teacher answers by source ID."""
from collections import Counter
import json
from pathlib import Path
from rmt.training_data import file_sha


def main():
    root=Path('artifacts/v0.0.5/corpus')
    source=[json.loads(x) for x in (root/'train.jsonl').read_text().split('\n') if x]
    teacher={x['id']:x for x in (json.loads(s) for s in (root/'teacher32_train.jsonl').read_text().split('\n') if s)}
    rows=[];counts=Counter();targets=Counter();replaced=0
    for original in source:
        row=dict(original)
        if row['id'] in teacher:
            t=teacher[row['id']]
            row.update({k:t[k] for k in ['messages','input_ids','labels','teacher_identity','verification']})
            row['answer_origin']='qwen3_32b';replaced+=1
        else:row['answer_origin']='public_source'
        rows.append(row);counts[row['domain']]+=1
        targets[row['domain']]+=sum(x!=-100 for x in row['labels'])
    assert len(rows)==len({x['id'] for x in rows})
    p=root/'mixed_train.jsonl'
    p.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows))
    report={'records':len(rows),'unique_ids':len(rows),'replaced_by_teacher':replaced,
        'domains':dict(counts),'targets_by_domain':dict(targets),'sha256':file_sha(p),
        'source_sha256':file_sha(root/'train.jsonl'),'teacher_sha256':file_sha(root/'teacher32_train.jsonl'),
        'policy':'Replace matching source answers with accepted 32B answers; retain broad source pool without domain oversampling. Natural source-bucket proportions, not a claimed 50/25/15/10 mixture.'}
    Path('reports/v0.0.5/mixed_corpus.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()

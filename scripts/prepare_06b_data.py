"""Retokenize audited source records; never reuse another model's token cache."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from transformers import AutoTokenizer
from rmt.training_data import tokenize_response, file_sha


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--base',default='/share/project/eai_pwm/models/Qwen/Qwen3-0.6B')
    p.add_argument('--source',default='artifacts/v0.0.4_dynamic_recurr/corpus')
    p.add_argument('--output',default='artifacts/v0.0.5/corpus')
    a=p.parse_args();root=Path(a.output);root.mkdir(parents=True,exist_ok=True)
    tok=AutoTokenizer.from_pretrained(a.base,local_files_only=True)
    manifest={'base':a.base,'template_sha256':hashlib.sha256(tok.chat_template.encode()).hexdigest(),
              'max_document_tokens':2048,'splits':{},
              'policy':'Preserve complete answers; exclude long documents, no token truncation. Historical source split identities retained.'}
    split_ids={}
    for name in ('train','teacher32_train','dev','halt_calibration','test'):
        source=Path(a.source)/(name+'.jsonl');rows=[];excluded=Counter()
        with source.open() as f:
            for line in f:
                row=json.loads(line)
                try:ids,labels=tokenize_response(tok,row['messages'])
                except ValueError:excluded['template']+=1;continue
                if len(ids)>2048:excluded['length']+=1;continue
                if not any(x!=-100 for x in labels[1:]):excluded['no_targets']+=1;continue
                row.update(input_ids=ids,labels=labels);rows.append(row)
        target=root/(name+'.jsonl')
        target.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
        split_ids[name]={r['id'] for r in rows}
        manifest['splits'][name]={'source_sha256':file_sha(source),'sha256':file_sha(target),
            'records':len(rows),'input_tokens':sum(len(r['input_ids']) for r in rows),
            'targets':sum(sum(x!=-100 for x in r['labels']) for r in rows),
            'domains':dict(Counter(r['domain'] for r in rows)),'excluded':dict(excluded)}
    for name in ('dev','halt_calibration','test'):
        assert not split_ids[name] & (split_ids['train']|split_ids['teacher32_train'])
    assert not split_ids['dev'] & split_ids['halt_calibration']
    assert not split_ids['dev'] & split_ids['test']
    assert not split_ids['test'] & split_ids['halt_calibration']
    manifest['split_overlap_check']='passed'
    Path('reports/v0.0.5/corpus.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(manifest,ensure_ascii=False),flush=True)


if __name__=='__main__':main()

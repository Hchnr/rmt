"""Revision-checked dataset-server pages, grouped splits and benchmark filtering."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import re
import time
import unicodedata
import requests
from transformers import AutoTokenizer
from rmt.training_data import tokenize_response
from rmt.evaluation.prepare import records

SOURCES=[
 ('general','HuggingFaceH4/ultrachat_200k','8049631c405ae6576f93f445c6b8166f76f5505a','default','train_sft',4000),
 ('math','open-r1/OpenR1-Math-220k','e4e141ec9dea9f8326f4d347be56105859b2bd68','default','train',3000),
 ('code','OpenCoder-LLM/opc-sft-stage1','1bcab575f5e2d1c1fd6652720418524c27b3d58b','largescale_diverse_instruct','train',2000),
 ('chinese','OpenCoder-LLM/opc-sft-stage1','1bcab575f5e2d1c1fd6652720418524c27b3d58b','filtered_infinity_instruct','train',2000)]


def normalize(text):
    return ' '.join(unicodedata.normalize('NFKC',text).casefold().split())


def main():
    root=Path('artifacts/v0.0.4_dynamic_recurr/corpus');root.mkdir(parents=True,exist_ok=True)
    def page(task):
        source,offset=task;domain,repo,rev,config,split,_=source
        path=root/'pages'/domain/f'{offset:06}.json';path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists():
            result=json.loads(path.read_text())
            if result['revision']!=rev:raise ValueError('Cached source revision mismatch')
            return result
        for attempt in range(3):
            try:
                response=requests.get('https://datasets-server.huggingface.co/rows',params={
                    'dataset':repo,'config':config,'split':split,'offset':offset,'length':100},timeout=(20,90))
                response.raise_for_status()
                if response.headers.get('x-revision')!=rev:raise ValueError('Dataset server revision does not match pinned source')
                data=response.json()
                result={'domain':domain,'repo':repo,'revision':rev,'config':config,'split':split,'offset':offset,'rows':data['rows']}
                path.write_text(json.dumps(result,ensure_ascii=False)+'\n');return result
            except Exception as error:
                if attempt==2:raise RuntimeError(f'{domain} page {offset}: {type(error).__name__}') from None
                time.sleep(2)
    tasks=[(source,offset) for source in SOURCES for offset in range(0,source[-1],100)]
    with concurrent.futures.ThreadPoolExecutor(8) as pool:
        pages=[]
        for result in pool.map(page,tasks):
            pages.append(result);print(json.dumps({'source':result['domain'],'page':result['offset']}),flush=True)
    benchmark_text=set();audit_sources=[]
    for path in sorted(Path('artifacts/v0.0.3/datasets/source').rglob('*')):
        if path.suffix not in ('.jsonl','.arrow','.parquet'):continue
        rows=records(path)
        for row in rows:
            for key in ('question','problem','prompt','question_content'):
                if isinstance(row.get(key),str):benchmark_text.add(normalize(row[key]))
        audit_sources.append({'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    # Normalized alphanumeric identity also catches punctuation-only changes.
    def skeleton(s):return ''.join(c for c in normalize(s) if c.isalnum())
    benchmark_skeleton={skeleton(x) for x in benchmark_text}
    tokenizer=AutoTokenizer.from_pretrained('/share/project/eai_pwm/models/Qwen/Qwen3-4B',local_files_only=True)
    split_rows={s:[] for s in ('train','halt_calibration','dev','test')};seen=set();excluded={};domain_counts={}
    def reject(reason):excluded[reason]=excluded.get(reason,0)+1
    for page_data in pages:
        domain=page_data['domain']
        for item in page_data['rows']:
            if item.get('truncated_cells'):reject('server_truncated_cells');continue
            row=item['row']
            if domain=='general':
                messages=row.get('messages',[])
                if len(messages)<2 or messages[0]['role']!='user' or messages[1]['role']!='assistant':reject('unsupported_dialogue');continue
                prompt,answer=messages[0]['content'],messages[1]['content']
            elif domain=='math':
                prompt=row['problem'];answer=row.get('solution') or ''
                if re.search(r'(?i)(aime|amc|math[_ -]?500|hendrycks|olympiadbench)',str(row.get('source',''))):reject('benchmark_source');continue
            else:prompt,answer=row.get('instruction',''),row.get('output','')
            if not isinstance(prompt,str) or not isinstance(answer,str) or not prompt.strip() or not answer.strip():reject('empty');continue
            if domain=='chinese' and not re.search('[\u4e00-\u9fff]',prompt):reject('non_chinese_fallback');continue
            key=skeleton(prompt)
            if key in seen:reject('duplicate_group');continue
            seen.add(key)
            if normalize(prompt) in benchmark_text or key in benchmark_skeleton:reject('benchmark_match');continue
            # Conservative high-overlap screen, bounded to stems long enough to be specific.
            if any(len(x)>=80 and (x in key or key in x) for x in benchmark_skeleton if len(key)>=80):reject('benchmark_containment');continue
            if '<think>' in answer or '</think>' in answer:reject('thinking_trace');continue
            messages=[{'role':'user','content':prompt},{'role':'assistant','content':answer}]
            try:ids,labels=tokenize_response(tokenizer,messages)
            except ValueError:reject('template');continue
            if len(ids)>8192 or len(ids)<32:reject('length');continue
            digest=hashlib.sha256(key.encode()).hexdigest();bucket=int(digest[:8],16)%10000
            split='train' if bucket<9000 else 'halt_calibration' if bucket<9300 else 'dev' if bucket<9600 else 'test'
            output={'id':digest,'prompt_sha256':digest,'domain':domain,'messages':messages,'input_ids':ids,'labels':labels,
                'source':page_data['repo'],'revision':page_data['revision'],'source_config':page_data['config'],
                'source_row':item['row_idx'],'reference_answer':row.get('answer'),'original_source':row.get('source',row.get('tag'))}
            split_rows[split].append(output);domain_counts[domain]=domain_counts.get(domain,0)+1
    manifest={'sources':SOURCES,'splits':{},'excluded':excluded,'domains':domain_counts,
        'benchmark_sources':audit_sources,'benchmark_stems':len(benchmark_text),
        'scope':'Pinned page revisions and stored hashes; exact/punctuation/containment screening, not semantic decontamination or pretraining guarantee',
        'fallback':'Infinity-Instruct Gen access returned HTTP 401; public OpenCoder filtered_infinity_instruct Chinese rows used explicitly',
        'template_sha256':hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
        'response_origin':'Original source answers, not yet Qwen3-32B distillation; single-turn complete pairs; hash-grouped split before teacher generation'}
    for split,rows in split_rows.items():
        rows.sort(key=lambda x:x['id']);path=root/(split+'.jsonl')
        path.write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows))
        manifest['splits'][split]={'records':len(rows),'input_tokens':sum(len(x['input_ids']) for x in rows),
            'targets':sum(sum(y!=-100 for y in x['labels']) for x in rows),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    manifest['pages']={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((root/'pages').rglob('*.json'))}
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    Path('reports/v0.0.4_dynamic_recurr/corpus_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'splits':manifest['splits'],'excluded':excluded,'domains':domain_counts}),flush=True)


if __name__=='__main__':main()

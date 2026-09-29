"""Pin upstream bytes and select records; EvalScope retains prompts and scoring."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import random
import requests
import time


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,default=str).encode()).hexdigest()


def fetch_ranges(url, path, size, expected_sha, workers=24):
    """Resume immutable large files in bounded chunks; check range and LFS hash."""
    chunk=4*1024**2
    def part(i):
        start=i*chunk;end=min(size,start+chunk)-1
        target=path.with_name(path.name+f'.part{i}')
        if target.exists() and target.stat().st_size==end-start+1:return target
        for attempt in range(4):
            try:
                response=requests.get(url,params={'range_start':start},
                    headers={'Range':f'bytes={start}-{end}'},timeout=(15,90))
                response.raise_for_status()
                if response.status_code!=206 or response.headers.get('Content-Range')!=f'bytes {start}-{end}/{size}' or len(response.content)!=end-start+1:
                    raise ValueError('Server did not honor byte range')
                target.write_bytes(response.content);return target
            except (requests.RequestException,ValueError) as error:
                if attempt==3:raise RuntimeError(f'Range {i} failed: {type(error).__name__}') from None
                time.sleep(1+attempt)
    with ThreadPoolExecutor(workers) as pool:
        parts=list(pool.map(part,range((size+chunk-1)//chunk)))
    staging=path.with_suffix(path.suffix+'.assembling');h=hashlib.sha256()
    with staging.open('wb') as out:
        for item in parts:
            block=item.read_bytes();h.update(block);out.write(block)
    if h.hexdigest()!=expected_sha:raise RuntimeError('Downloaded LFS SHA256 mismatch')
    staging.replace(path)
    for item in parts:item.unlink()


def fetch(spec,name,root):
    path=root/'source'/spec['repo']/spec['revision']/name
    if not path.exists():
        path.parent.mkdir(parents=True,exist_ok=True)
        url=f"https://huggingface.co/datasets/{spec['repo']}/resolve/{spec['revision']}/{name}"
        if name=='test5.jsonl' and spec['repo']=='livecodebench/code_generation_lite':
            fetch_ranges(url,path,spec['file_size'],spec['file_sha256'])
            return path
        for attempt in range(3):
            try:
                with requests.get(url,timeout=(15,60),stream=True) as response:
                    response.raise_for_status()
                    with path.with_suffix(path.suffix+'.part').open('wb') as out:
                        for chunk in response.iter_content(1024*1024):out.write(chunk)
                path.with_suffix(path.suffix+'.part').replace(path);break
            except requests.RequestException as error:
                if attempt==2:raise RuntimeError(f'Download failed: {spec["repo"]}/{name}: {type(error).__name__}') from None
                time.sleep(1+attempt)
    if name=='test5.jsonl' and hashlib.sha256(path.read_bytes()).hexdigest()!=spec['file_sha256']:
        raise RuntimeError('Cached LCB LFS SHA256 mismatch')
    return path


def records(path):
    if path.suffix=='.arrow':
        from datasets import Dataset
        return Dataset.from_file(str(path)).to_list()
    if path.suffix=='.parquet':
        import pyarrow.parquet as pq
        return pq.read_table(path).to_pylist()
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def prepare(name,spec,config,root):
    sources=[];groups={};rng=random.Random(config['seed'])
    def read(filename):
        path=fetch(spec,filename,root)
        sources.append({'file':filename,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
        return records(path)
    if name=='mmlu_redux':
        groups={s:read(f'{s}/data-00000-of-00001.arrow') for s in spec['subsets']}
    elif name=='ceval':
        groups={s:read(f'{s}/val-00000-of-00001.parquet') for s in spec['subsets']}
    elif name=='ifeval':groups={'default':read('ifeval_input_data.jsonl')}
    elif name=='math_500':
        raw=read('test.jsonl');groups={s:[x for x in raw if f"Level {x['level']}"==s] for s in spec['subsets']}
    else:
        # Historical quick regression uses only test5, NOT the release_v5 union.
        # At this revision it ends 2025-01-04; see v0.0.4 coverage audit.
        raw=[]
        for file in ['test5.jsonl']:raw.extend(read(file))
        groups={'release_v5':[x for x in raw if spec['start_date']<=x['contest_date'][:10]<=spec['end_date']]}
    manifests={}
    for phase,total in [('pilot',config['pilot_samples_per_benchmark']),('representative',config['samples_per_benchmark'])]:
        dest=root/phase/name;dest.mkdir(parents=True,exist_ok=True)
        selected=[];math_rows=[]
        for j,(subset,rows) in enumerate(groups.items()):
            # Hash ranking is stable across run order and pilot/formal sizes.
            ranked=sorted(enumerate(rows),key=lambda pair:digest([config['seed'],subset,pair[1]]))
            count=total//len(groups)+(j<total%len(groups))
            if len(ranked)<count:raise ValueError(f'Insufficient {name}/{subset} records')
            chosen=ranked[:count]
            chosen_rows=[row for _,row in chosen]
            if name=='math_500':math_rows.extend(chosen_rows)
            else:
                (dest/f'{subset}_{spec["split"]}.jsonl').write_text(''.join(json.dumps(row,ensure_ascii=False,default=str)+'\n' for row in chosen_rows))
            selected.extend({'subset':subset,'source_index':index,'record_sha256':digest(row),
                'source_id':str(row.get('question_id',row.get('unique_id',row.get('key',row.get('id',index)))))} for index,row in chosen)
        if math_rows:(dest/'default_test.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in math_rows))
        manifests[phase]={'benchmark':name,'source':spec,'files':sources,'selection':selected,'count':len(selected),'seed':config['seed']}
        (dest/'manifest.json').write_text(json.dumps(manifests[phase],indent=2,ensure_ascii=False)+'\n')
    print(name,{k:v['count'] for k,v in manifests.items()},flush=True)
    return name,manifests


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/eval/quick_non_thinking.json');p.add_argument('--root',default='artifacts/v0.0.3/datasets')
    args=p.parse_args();config=json.loads(Path(args.config).read_text());root=Path(args.root)
    with ThreadPoolExecutor(5) as pool:
        futures=[pool.submit(prepare,name,spec,config,root) for name,spec in config['benchmarks'].items()]
        result=dict(f.result() for f in futures)
    out=Path('reports/v0.0.3/dataset_manifest.json');out.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')


if __name__=='__main__':main()

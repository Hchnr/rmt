"""Fetch bounded shards of pinned, separately attributed instruction sources."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import requests

SOURCES=[
 ('general','HuggingFaceH4/ultrachat_200k','8049631c405ae6576f93f445c6b8166f76f5505a','data/train_sft-'),
 ('math','open-r1/OpenR1-Math-220k','e4e141ec9dea9f8326f4d347be56105859b2bd68','data/train-'),
 ('code','OpenCoder-LLM/opc-sft-stage1','1bcab575f5e2d1c1fd6652720418524c27b3d58b','data/largescale_diverse_instruct-'),
 ('chinese','BAAI/Infinity-Instruct','bddc39a8feadbd679c30623197f4e736b7e75b48','Gen/train-')]


def main():
    root=Path('artifacts/v0.0.4_dynamic_recurr/corpus/source');root.mkdir(parents=True,exist_ok=True)
    def fetch(source):
        name,repo,revision,prefix=source
        r=requests.get(f'https://huggingface.co/api/datasets/{repo}/revision/{revision}',timeout=30);r.raise_for_status()
        candidates=sorted(x['rfilename'] for x in r.json()['siblings'] if x['rfilename'].startswith(prefix) and x['rfilename'].endswith('.parquet'))
        if not candidates:raise ValueError(f'No matching shard for {name}')
        files={}
        for remote in ['README.md',candidates[0]]:
            path=root/name/Path(remote).name;path.parent.mkdir(parents=True,exist_ok=True)
            if not path.exists():
                with requests.get(f'https://huggingface.co/datasets/{repo}/resolve/{revision}/{remote}',stream=True,timeout=(30,180)) as resp:
                    if not resp.ok:raise RuntimeError(f'{name} HTTP {resp.status_code}')
                    partial=Path(str(path)+'.part')
                    with partial.open('wb') as f:
                        for chunk in resp.iter_content(4*1024*1024):f.write(chunk)
                    partial.replace(path)
            h=hashlib.sha256()
            with path.open('rb') as f:
                for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
            files[remote]={'path':str(path),'sha256':h.hexdigest(),'bytes':path.stat().st_size}
        result={'domain':name,'repo':repo,'revision':revision,'files':files,'scope':'one pinned training shard; source responses are not Qwen distillation'}
        print(json.dumps(result),flush=True);return result
    with concurrent.futures.ThreadPoolExecutor(4) as pool:results=list(pool.map(fetch,SOURCES))
    Path('reports/v0.0.4_dynamic_recurr/corpus_sources.json').write_text(json.dumps(results,indent=2)+'\n')


if __name__=='__main__':main()

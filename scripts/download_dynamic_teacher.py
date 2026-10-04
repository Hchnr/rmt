"""Pinned teacher download with streamed SHA256 verification and bounded retries."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import time
import requests


def main():
    repo='Qwen/Qwen3-32B'
    revision='9216db5781bf21249d130ec9da846c4624c16137'
    root=Path('artifacts/v0.0.4_dynamic_recurr/teacher/Qwen3-32B')
    root.mkdir(parents=True,exist_ok=True)
    response=requests.get(f'https://huggingface.co/api/models/{repo}/revision/{revision}?blobs=true',timeout=30)
    response.raise_for_status()
    files=[x for x in response.json()['siblings'] if x['rfilename'].endswith(('.json','.safetensors','.txt')) or x['rfilename'] in ('LICENSE','README.md')]
    def download(item):
        name=item['rfilename'];path=root/name;path.parent.mkdir(parents=True,exist_ok=True)
        expected=item.get('lfs',{}).get('sha256')
        for attempt in range(3):
            try:
                if not path.exists():
                    with requests.get(f'https://huggingface.co/{repo}/resolve/{revision}/{name}',stream=True,timeout=(30,120)) as r:
                        r.raise_for_status()
                        staging=Path(str(path)+'.part')
                        with staging.open('wb') as f:
                            for chunk in r.iter_content(8*1024*1024):f.write(chunk)
                        staging.replace(path)
                h=hashlib.sha256()
                with path.open('rb') as f:
                    for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
                actual=h.hexdigest()
                if expected and actual!=expected:
                    path.unlink();raise ValueError('Checksum mismatch')
                print(json.dumps({'file':name,'bytes':path.stat().st_size,'status':'verified'}),flush=True)
                return name,{'sha256':actual,'bytes':path.stat().st_size}
            except Exception as error:
                print(json.dumps({'file':name,'attempt':attempt+1,'error_type':type(error).__name__}),flush=True)
                time.sleep(2)
        raise RuntimeError(f'Download failed after bounded retries: {name}')
    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        result=dict(pool.map(download,files))
    manifest={'repo':repo,'revision':revision,'files':result,'status':'verified'}
    (root/'download_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    Path('reports/v0.0.4_dynamic_recurr/teacher_source.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':main()

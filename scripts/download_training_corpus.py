"""Download a pinned human instruction corpus; never read proxy credentials."""
import hashlib
import json
from pathlib import Path
import requests
import pyarrow.parquet as pq
repo='HuggingFaceH4/no_robots';revision='e6f9a4ac5c37faeb744ba9ecf0473184d7f8105b'
root=Path('artifacts/v0.0.4/corpus/source');root.mkdir(parents=True,exist_ok=True)
manifest={'repo':repo,'revision':revision,'license':'CC-BY-NC-4.0','use':'research stability pilot, English instructions; not final distillation mix','files':{}}
for name in ['README.md','data/train-00000-of-00001.parquet','data/test-00000-of-00001.parquet']:
 path=root/Path(name).name
 if not path.exists():
  with requests.get(f'https://huggingface.co/datasets/{repo}/resolve/{revision}/{name}',timeout=(15,120),stream=True) as r:
   r.raise_for_status()
   staging=path.with_suffix(path.suffix+'.part')
   with staging.open('wb') as f:
    for chunk in r.iter_content(1024*1024):f.write(chunk)
   staging.replace(path)
 manifest['files'][name]={'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size}
 if path.suffix=='.parquet':
  rows=pq.read_table(path).to_pylist();split='train' if 'train' in name else 'dev'
  (root.parent/f'{split}_raw.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows))
  manifest['files'][name]['rows']=len(rows)
Path('reports/v0.0.4/corpus_source.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(manifest))

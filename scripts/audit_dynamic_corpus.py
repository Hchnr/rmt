"""Verify frozen splits without inspecting held-out model scores."""
import hashlib
import json
from pathlib import Path
import re
import unicodedata
root=Path('artifacts/v0.0.4_dynamic_recurr/corpus');manifest=json.loads(Path('reports/v0.0.4_dynamic_recurr/corpus_manifest.json').read_text())
ids=set();prompts=set();records={}
for split,spec in manifest['splits'].items():
 path=root/(split+'.jsonl');assert hashlib.sha256(path.read_bytes()).hexdigest()==spec['sha256']
 rows=[json.loads(x) for x in path.read_text().splitlines()];assert len(rows)==spec['records']
 assert sum(len(x['input_ids']) for x in rows)==spec['input_tokens']
 assert sum(sum(y!=-100 for y in x['labels']) for x in rows)==spec['targets']
 for row in rows:
  prompt=unicodedata.normalize('NFKC',row['messages'][0]['content']).casefold()
  key=''.join(c for c in prompt if c.isalnum())
  assert row['id'] not in ids and key not in prompts
  ids.add(row['id']);prompts.add(key)
  assert len(row['input_ids'])==len(row['labels'])
  assert all(y==-100 or y==x for x,y in zip(row['input_ids'],row['labels']))
  assert any(y!=-100 for y in row['labels'][1:])
 records[split]=len(rows)
r={'status':'passed','records':records,'checks':['pinned full-file SHA256','counts/input/target totals','cross-split unique IDs and normalized alphanumeric prompts','label/ID equality on supervised positions'],
 'scope':'Exact normalized groups, not semantic paraphrase decontamination. No held-out model scores read.'}
Path('reports/v0.0.4_dynamic_recurr/corpus_audit.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))

"""Freeze instruction train/dev tokens without benchmark-driven selection."""
import hashlib
import json
from pathlib import Path
from transformers import AutoTokenizer
from rmt.training_data import tokenize_response,file_sha
root=Path('artifacts/v0.0.4/corpus')
tokenizer=AutoTokenizer.from_pretrained('/share/project/eai_pwm/models/Qwen/Qwen3-4B',local_files_only=True)
manifest={'source':json.loads(Path('reports/v0.0.4/corpus_source.json').read_text()),
 'protocol':'Final assistant response targets only; native non-thinking chat template; no benchmark data; chunk boundaries reset positions and mask first target',
 'max_document_tokens':8192,'splits':{},'template_sha256':hashlib.sha256(tokenizer.chat_template.encode()).hexdigest()}
seen=set()
for split in ['train','dev']:
 raw=[json.loads(x) for x in (root/f'{split}_raw.jsonl').read_text().split('\n') if x.strip()]
 rows=[];excluded={'duplicate':0,'too_long':0,'unsupported':0}
 for row in sorted(raw,key=lambda x:hashlib.sha256(x['prompt_id'].encode()).hexdigest()):
  # Exclude duplicate initial prompts across splits, even if responses differ.
  key=hashlib.sha256(row['prompt'].strip().encode()).hexdigest()
  if key in seen:excluded['duplicate']+=1;continue
  seen.add(key)
  try:ids,labels=tokenize_response(tokenizer,row['messages'])
  except ValueError:excluded['unsupported']+=1;continue
  if len(ids)>8192:excluded['too_long']+=1;continue
  rows.append({'id':row['prompt_id'],'prompt_sha256':key,'input_ids':ids,'labels':labels})
 path=root/f'{split}.jsonl';path.write_text(''.join(json.dumps(x)+'\n' for x in rows))
 manifest['splits'][split]={'records':len(rows),'input_tokens':sum(len(x['input_ids']) for x in rows),
  'targets':sum(sum(y!=-100 for y in x['labels']) for x in rows),'sha256':file_sha(path),'excluded':excluded}
 if split=='dev':
  pilot=root/'dev_pilot.jsonl';pilot.write_text(''.join(json.dumps(x)+'\n' for x in rows[:32]))
  manifest['dev_pilot']={'records':min(32,len(rows)),'sha256':file_sha(pilot),'selection':'first 32 hash-ranked held-out prompts, fixed before training'}
(root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
Path('reports/v0.0.4/corpus_tokens.json').write_text(json.dumps(manifest,indent=2)+'\n');print(manifest['splits'])

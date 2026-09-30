"""Post-hoc exact normalized question overlap audit against available eval pools.

No dataset/checkpoint changes or semantic decontamination claims.
"""
import hashlib
import json
from pathlib import Path
import unicodedata
from rmt.evaluation.prepare import records

def normalize(text):return ' '.join(unicodedata.normalize('NFKC',text).casefold().split())
corpus=Path('artifacts/v0.0.4/corpus')
retained={json.loads(x)['id'] for x in (corpus/'train.jsonl').open()}
users={};turns=0
for row in map(json.loads,(corpus/'train_raw.jsonl').open()):
 if row['prompt_id'] not in retained:continue
 for index,message in enumerate(row['messages']):
  if message['role']=='user':
   key=normalize(message['content']);users.setdefault(key,[]).append({'id':row['prompt_id'],'turn':index});turns+=1
sources=[];matches=[]
for path in sorted(Path('artifacts/v0.0.3/datasets/source').rglob('*')):
 if path.suffix not in ['.jsonl','.arrow','.parquet']:continue
 rows=records(path);count=0
 for index,row in enumerate(rows):
  for field in ['question','problem','prompt','question_content']:
   value=row.get(field)
   if isinstance(value,str) and value.strip():
    count+=1;key=normalize(value)
    if key in users:matches.append({'source':str(path),'row':index,'field':field,'question_sha256':hashlib.sha256(key.encode()).hexdigest(),'train_matches':users[key]})
 sources.append({'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'records':len(rows),'question_fields':count})
result={'retained_train_records':len(retained),'train_user_turns':turns,'normalization':'NFKC, casefold, collapse whitespace',
        'sources':sources,'matches':matches,'scope':'Available pinned IFEval/MATH files and existing Redux/C-Eval/LCB source pools only; exact question/stem text, no semantic or pretraining contamination guarantee; audit does not alter frozen experiments'}
Path('reports/v0.0.4/training_overlap_audit.json').write_text(json.dumps(result,indent=2)+'\n')
print({'source_files':len(sources),'question_fields':sum(x['question_fields'] for x in sources),'matches':len(matches)})

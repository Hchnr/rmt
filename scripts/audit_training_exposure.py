"""Replay pack membership to report source IDs and real token exposure, without GPUs."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import yaml
p=argparse.ArgumentParser();p.add_argument('--configs',nargs='+',required=True);p.add_argument('--world-size',type=int,default=4);p.add_argument('--output',required=True);a=p.parse_args()
reports=[]
for name in a.configs:
 c=yaml.safe_load(Path(name).read_text());path=Path(c['train_data']);rows=[json.loads(x) for x in path.read_text().splitlines()]
 random.Random(c.get('seed',17)).shuffle(rows);length=c['sequence_length'];packs=[];current=[];size=0
 for row in rows:
  for start in range(0,len(row['input_ids']),length):
   ids=row['input_ids'][start:start+length];labels=row['labels'][start:start+length].copy();labels[0]=-100
   targets=sum(x!=-100 for x in labels[1:])
   if not targets:continue
   if size+len(ids)>length:packs.append(current);current=[];size=0
   current.append({'id':row['id'],'offset':start,'input_tokens':len(ids),'targets':targets});size+=len(ids)
 if current:packs.append(current)
 visited=[packs[i%len(packs)] for i in range(c['steps']*a.world_size)];chunks=[x for pack in visited for x in pack]
 counts=Counter(x['id'] for x in chunks);tokens=sum(x['input_tokens'] for x in chunks);targets=sum(x['targets'] for x in chunks)
 report_path=Path(c['report']);verified=False
 if report_path.exists():
  r=json.loads(report_path.read_text());assert len(r['steps'])==c['steps']
  assert sum(x['input_tokens'] for x in r['steps'])==tokens
  assert sum(x['targets'] for x in r['steps'])==targets
  verified=True
 reports.append({'config':name,'data_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'world_size':a.world_size,
  'total_records':len(rows),'packed_batches':len(packs),'scheduled_batches':len(visited),'input_tokens':tokens,'targets':targets,
  'unique_source_ids_seen':len(counts),'source_chunk_visits':dict(sorted(counts.items())),
  'continued_chunks_without_original_prefix':sum(x['offset']>0 for x in chunks),'matches_completed_training_report':verified})
Path(a.output).write_text(json.dumps({'rows':reports,'scope':'Deterministic scheduled pack/chunk exposure; source IDs may span multiple chunks. Train report totals independently checked when available.'},indent=2)+'\n')
for r in reports:print({k:r[k] for k in ['config','input_tokens','targets','unique_source_ids_seen','matches_completed_training_report']})

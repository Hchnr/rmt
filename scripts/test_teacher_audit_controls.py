"""Synthetic controls for answer filtering and fail-closed provenance checks."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
with tempfile.TemporaryDirectory(prefix='rmt_teacher_audit_') as directory:
 root=Path(directory);answers=root/'answers';answers.mkdir();rows=[];generated=[]
 for i,(text,finish) in enumerate([(r'The answer is $\boxed{42}$.','stop'),(r'The answer is $\boxed{43}$.','stop'),('This response was cut off before finishing.','length')]):
  messages=[{'role':'user','content':f'Control {i}: What is 6 times 7?'},{'role':'assistant','content':'42'}]
  rows.append({'id':str(i),'domain':'math','messages':messages,'input_ids':[1,2,3],'labels':[-100,2,3]})
  generated.append({'id':str(i),'domain':'math','messages':messages[:1],'reference_answer':'42','text':text,'finish_reason':finish})
 source=root/'source.jsonl';source.write_text(''.join(json.dumps(x)+'\n' for x in rows))
 identity={'model':'synthetic-test-only','data_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
  'selection_ids':['0','1','2'],'rank':0,'world_size':1}
 (answers/'rank_0_identity.json').write_text(json.dumps(identity))
 output=answers/'rank_0.jsonl';output.write_text(''.join(json.dumps(x)+'\n' for x in generated))
 (answers/'rank_0_summary.json').write_text(json.dumps({'status':'completed','output_sha256':hashlib.sha256(output.read_bytes()).hexdigest()}))
 cmd=[sys.executable,'scripts/prepare_teacher_distillation.py','--source',str(source),'--answers',str(answers),'--output',str(root/'accepted.jsonl'),'--report',str(root/'report.json')]
 run=subprocess.run(cmd,capture_output=True,text=True)
 if run.returncode:raise RuntimeError(run.stderr)
 report=json.loads((root/'report.json').read_text());assert report['accepted']==1
 assert report['excluded']=={'math_unverified':1,'truncated':1}
 assert [json.loads(x)['id'] for x in (root/'matched_source_train.jsonl').read_text().splitlines()]==['0']
 output.write_text(output.read_text()+'\n')
 run=subprocess.run(cmd,capture_output=True,text=True);assert run.returncode and 'Teacher output hash mismatch' in run.stderr
 print('Synthetic controls passed: correct/wrong math, truncation, matching source IDs, modified-output rejection')

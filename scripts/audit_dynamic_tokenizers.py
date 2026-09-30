"""Check complete vocabulary mappings and rendered prefixes before distillation/eval."""
import argparse
import hashlib
import json
from pathlib import Path
from transformers import AutoTokenizer
p=argparse.ArgumentParser();p.add_argument('--candidate',required=True);p.add_argument('--output',required=True);a=p.parse_args()
base=AutoTokenizer.from_pretrained('/share/project/eai_pwm/models/Qwen/Qwen3-4B',local_files_only=True)
candidate=AutoTokenizer.from_pretrained(a.candidate,local_files_only=True)
messages=[[{'role':'user','content':x}] for x in ['Hello!','证明勾股定理。',r'Find $x^2+1=2$.','Write a Python function.\n```python\ndef f(x):\n    return x\n```','Spaces  tabs\tand Unicode 🧪']]
rows=[]
for value in messages:
 left=base.apply_chat_template(value,tokenize=False,add_generation_prompt=True,enable_thinking=False)
 right=candidate.apply_chat_template(value,tokenize=False,add_generation_prompt=True,enable_thinking=False)
 rows.append({'messages':value,'template_equal':left==right,'token_ids_equal':base.encode(left,add_special_tokens=False)==candidate.encode(right,add_special_tokens=False)})
r={'candidate':a.candidate,'vocabulary_mapping_equal':base.get_vocab()==candidate.get_vocab(),
   'special_tokens_equal':base.special_tokens_map==candidate.special_tokens_map,'prefixes':rows,
   'template_sha256':hashlib.sha256(candidate.chat_template.encode()).hexdigest()}
r['status']='passed' if r['vocabulary_mapping_equal'] and r['special_tokens_equal'] and all(x['template_equal'] and x['token_ids_equal'] for x in rows) else 'different_sequence_distillation_only'
Path(a.output).write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n');print(r['status'])

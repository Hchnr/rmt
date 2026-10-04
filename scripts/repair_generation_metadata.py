"""Create an immutable HF variant with reference generation metadata only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True)
p.add_argument('--reference',default='/share/project/eai_pwm/models/Qwen/Qwen3-4B');a=p.parse_args()
source=Path(a.source);output=Path(a.output);reference=Path(a.reference)/'generation_config.json'
assert (source/'config.json').is_file() and reference.is_file()
old=json.loads((source/'generation_config.json').read_text());new=json.loads(reference.read_text())
output.mkdir(parents=True,exist_ok=False)
for file in source.iterdir():
 if file.is_file() and file.name!='generation_config.json':os.link(file,output/file.name)
(output/'generation_config.json').write_bytes(reference.read_bytes())
weights=[p.name for p in source.glob('*.safetensors')]
assert weights and all(os.path.samefile(source/name,output/name) for name in weights)
result={'source':str(source),'output':str(output),'reference':str(reference),'old_generation_config':old,
 'new_generation_config':new,'generation_config_sha256':hashlib.sha256(reference.read_bytes()).hexdigest(),
 'unchanged_weight_files':weights,'weights_same_inode':True,'model_config_same_inode':os.path.samefile(source/'config.json',output/'config.json'),
 'scope':'Generation metadata correction only. Model weights, forward config, routing, halting thresholds and teacher-forcing scores are unchanged.'}
(output/'generation_repair.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

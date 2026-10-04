"""Create an immutable HF config variant sharing unchanged weight files by hard link."""
import argparse
import hashlib
import json
import os
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);p.add_argument('--calibration',required=True);a=p.parse_args()
source=Path(a.source);output=Path(a.output);calibration=Path(a.calibration)
report=json.loads(calibration.read_text());selected=report['selected']
if report['status']!='calibrated' or selected is None:raise ValueError('No budget-qualified calibration candidate')
if Path(report['model']).resolve()!=source.resolve():raise ValueError('Calibration model mismatch')
config=json.loads((source/'config.json').read_text());config['halting_policy']=report['policy']
config['halt_threshold']=config['halt_relative_threshold']=selected['hidden_threshold'];config['halt_probability_threshold']=selected['threshold']
output.mkdir(parents=True,exist_ok=False)
for path in source.iterdir():
 if path.is_file() and path.name!='config.json':os.link(path,output/path.name)
(output/'config.json').write_text(json.dumps(config,indent=2)+'\n')
identity={'source':str(source),'output':str(output),'calibration':str(calibration),
 'calibration_sha256':hashlib.sha256(calibration.read_bytes()).hexdigest(),'selected':selected,
 'config_sha256':hashlib.sha256((output/'config.json').read_bytes()).hexdigest(),'weights':'unchanged source files, hard links; config only recalibrated'}
(output/'variant_identity.json').write_text(json.dumps(identity,indent=2)+'\n');print(json.dumps(identity))

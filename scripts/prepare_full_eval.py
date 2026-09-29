"""Reuse pinned upstream IFEval/MATH-500 bytes, with full record manifests."""
import json
from pathlib import Path
from rmt.evaluation.prepare import records,digest
from rmt.evaluation.run import file_hash
cfg=json.loads(Path('configs/eval/quick_non_thinking.json').read_text())
cfg['name']='v004_full_non_thinking';cfg['generation']['max_tokens']=32768
cfg['benchmarks']={k:v for k,v in cfg['benchmarks'].items() if k in ['ifeval','math_500']}
cfg['deviations']=['Full IFEval and MATH-500 only; other report benchmarks not covered',
 'Official EvalScope 1.0.0 prompts/scorers; exact paper prompt parity not established',
 'One request-specific sample seed; not identical to unpublished paper RNG']
manifest={}
for name,spec in cfg['benchmarks'].items():
 filename='ifeval_input_data.jsonl' if name=='ifeval' else 'test.jsonl'
 source=Path('artifacts/v0.0.3/datasets/source')/spec['repo']/spec['revision']/filename
 rows=records(source);dest=Path('artifacts/v0.0.4/datasets/full')/name;dest.mkdir(parents=True,exist_ok=True)
 (dest/f'default_{spec["split"]}.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows))
 value={'benchmark':name,'source':spec,'files':[{'file':filename,'sha256':file_hash(source)}],
  'selection':[{'source_index':i,'record_sha256':digest(x)} for i,x in enumerate(rows)],'count':len(rows),'seed':cfg['seed'],'scope':'entire pinned source file'}
 (dest/'manifest.json').write_text(json.dumps(value,indent=2)+'\n');manifest[name]=value
Path('configs/eval/v004_full_non_thinking.json').write_text(json.dumps(cfg,indent=2)+'\n')
Path('reports/v0.0.4/full_dataset_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print({k:v['count'] for k,v in manifest.items()})

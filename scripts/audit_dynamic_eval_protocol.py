"""Check the reusable native baseline against the new generation/scoring protocol."""
import hashlib
import json
from pathlib import Path
from rmt.evaluation.run import model_fingerprint
from rmt.runtime import versions
profile=json.loads(Path('configs/eval/v004_dynamic_full_non_thinking.json').read_text());old=json.loads(Path('reports/v0.0.4/full_baseline.json').read_text())
current=versions();rows=[]
for benchmark,value in old['results'].items():
 work=Path(value['work']);provenance=json.loads((work/'provenance.json').read_text());metadata=provenance['metadata']
 generation_equal=provenance['protocol']['generation']==profile['generation']
 assert generation_equal and provenance['protocol']['benchmarks'][benchmark]==profile['benchmarks'][benchmark]
 selection=json.loads((Path('artifacts/v0.0.4/datasets/full')/benchmark/'manifest.json').read_text())
 assert provenance['selection']==selection
 assert metadata['weights']==model_fingerprint(metadata['model'])
 for key in ['python','torch','cuda_runtime','transformers','tf32_override','nvidia_tf32_override']:
  assert metadata['engine_environment'][key]==current[key],(key,metadata['engine_environment'][key],current[key])
 rows.append({'benchmark':benchmark,'work':str(work),'generation_equal':generation_equal,'selection_equal':True,'native_weight_hashes_equal':True,
   'engine_versions_equal':True,'baseline_scoring':'current official math scorer, all 500 unchanged' if benchmark=='math_500' else 'new per-record seeded official scoring of the same 541 answers; historical scores retained separately'})
report={'status':'passed','profile':'configs/eval/v004_dynamic_full_non_thinking.json','rows':rows,
 'scope':'Same frozen generation settings, samples, native weights and engine versions. No claim of identical microbatch layouts or exact paper prompts/RNG.'}
Path('reports/v0.0.4_dynamic_recurr/eval_protocol_audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)

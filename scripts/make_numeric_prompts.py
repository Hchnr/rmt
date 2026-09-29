"""Render real EvalScope benchmark messages without loading a model."""
import json
from pathlib import Path
from evalscope import TaskConfig
from evalscope.api.registry import get_benchmark
from evalscope.api.benchmark import DefaultDataAdapter
from rmt.evaluation.prepare import digest
original=DefaultDataAdapter.load_from_disk
DefaultDataAdapter.load_from_disk=lambda self,use_local_loader=False:original(self,use_local_loader=True)
protocol=json.loads(Path('configs/eval/quick_non_thinking.json').read_text());rows=[]
for name,spec in protocol['benchmarks'].items():
    data=Path('artifacts/v0.0.3/datasets/pilot')/name
    config=TaskConfig(datasets=[name],dataset_args={name:{'dataset_id':str(data.resolve()),'subset_list':spec['subsets'],'few_shot_num':0}})
    adapter=get_benchmark(name,config);datasets=adapter.load_dataset()
    selected=[sample for samples in datasets.values() for sample in samples][:4]
    for sample in selected:
        messages=[{'role':m.role,'content':m.content} for m in sample.input]
        rows.append({'benchmark':name,'messages':messages,'sha256':digest(messages)})
Path('artifacts/v0.0.3/numeric_prompts.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')
Path('reports/v0.0.3/numeric_prompt_manifest.json').write_text(json.dumps([{k:v for k,v in row.items() if k!='messages'} for row in rows],indent=2)+'\n')
print(len(rows))

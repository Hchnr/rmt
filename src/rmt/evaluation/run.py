"""EvalScope orchestration with immutable selection and fail-closed score auditing."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
import time
import requests
from .prepare import digest


def file_hash(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def model_fingerprint(path):
    path=Path(path)
    files=sorted([*path.glob('*.safetensors'),*path.glob('*.json'),*path.glob('*.jinja')])
    cache=Path('artifacts/v0.0.3/fingerprints');cache.mkdir(parents=True,exist_ok=True)
    stats={str(p):[p.stat().st_size,p.stat().st_mtime_ns] for p in files}
    target=cache/(digest(stats)+'.json')
    if target.exists():return json.loads(target.read_text())
    value={str(p.name):file_hash(p) for p in files};target.write_text(json.dumps(value));return value


def validate_generation_termination(metadata, protocol):
    expected=protocol.get('expected_eos_token_ids')
    if expected is not None:
        actual=metadata.get('generation_termination',{}).get('eos_token_ids')
        if actual is None or sorted(actual)!=sorted(expected):
            raise ValueError(f'EOS policy mismatch: expected {expected}, got {actual}')


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',choices=['rmt','qwen'],required=True)
    p.add_argument('--port',type=int,required=True);p.add_argument('--benchmark',required=True)
    p.add_argument('--phase',choices=['pilot','representative','full'],default='pilot')
    p.add_argument('--data-root',default='artifacts/v0.0.3/datasets');p.add_argument('--output-root',default='artifacts/v0.0.3/eval')
    p.add_argument('--config',default='configs/eval/quick_non_thinking.json');p.add_argument('--thinking',action='store_true')
    p.add_argument('--eval-batch-size',type=int,default=4)
    args=p.parse_args()
    if args.eval_batch_size<1:raise ValueError('Positive evaluation concurrency required')
    protocol=json.loads(Path(args.config).read_text());spec=protocol['benchmarks'][args.benchmark]
    data=Path(args.data_root)/args.phase/args.benchmark
    manifest=json.loads((data/'manifest.json').read_text())
    # Refuse changed local samples even if manifest and cache directory remain.
    hashes=[]
    for path in sorted(data.glob('*.jsonl')):
        hashes.extend(digest(json.loads(line)) for line in path.read_text().splitlines())
    if sorted(hashes)!=sorted(x['record_sha256'] for x in manifest['selection']):raise RuntimeError('Selection manifest mismatch')
    base=f'http://127.0.0.1:{args.port}'
    session=requests.Session();session.trust_env=False
    metadata=session.get(base+'/metadata',timeout=10).json()
    validate_generation_termination(metadata,protocol)
    metadata['weights']=model_fingerprint(metadata['model'])
    code={str(p):file_hash(p) for p in sorted(Path('src/rmt').rglob('*.py'))}
    generation=protocol['generation'].copy()
    if args.phase=='pilot':generation['max_tokens']=protocol.get('pilot_max_tokens',generation['max_tokens'])
    if args.thinking:generation={**generation,'extra_body':{'chat_template_kwargs':{'enable_thinking':True}}}
    scoring={'ifeval_score_seed':protocol.get('ifeval_score_seed') if args.benchmark=='ifeval' else None}
    execution={'eval_batch_size':args.eval_batch_size}
    key=digest({'execution':execution,'scoring':scoring,'model':metadata,'selection':manifest,'generation':generation,'code':code,'evalscope':'1.0.0'})
    work=Path(args.output_root)/args.phase/args.model/args.benchmark/key[:16]
    work.mkdir(parents=True,exist_ok=True)
    lock=threading.Lock()
    # Requests go to loopback only; do not inherit proxy routing for localhost.
    os.environ['NO_PROXY']='localhost,127.0.0.1';os.environ['no_proxy']=os.environ['NO_PROXY']
    os.environ['NLTK_DATA']=str(Path('artifacts/v0.0.3/nltk_data').resolve())
    from evalscope import TaskConfig,run_task
    from evalscope.api.benchmark import DefaultDataAdapter
    from evalscope.models.openai_compatible import OpenAICompatibleAPI
    from . import sandbox
    from .protocol import request_seed
    sandbox.install()
    if scoring['ifeval_score_seed'] is not None:
        from .ifeval_rng import install as install_ifeval_rng
        install_ifeval_rng(scoring['ifeval_score_seed'])
    # The official adapter's default local branch expects a HF builder. Our
    # pinned JSONL snapshots use its existing LocalDataLoader instead.
    original_load=DefaultDataAdapter.load_from_disk
    DefaultDataAdapter.load_from_disk=lambda self,use_local_loader=False:original_load(self,use_local_loader=True)
    original_generate=OpenAICompatibleAPI.generate
    def generate(self,input,tools,tool_choice,config):
        sample_seed=request_seed(input,protocol['seed'])
        return original_generate(self,input,tools,tool_choice,config.model_copy(update={'seed':sample_seed}))
    OpenAICompatibleAPI.generate=generate
    def response(self,value):
        with lock:
            with (work/'responses.jsonl').open('a') as f:f.write(json.dumps(value,ensure_ascii=False)+'\n')
    OpenAICompatibleAPI.on_response=response
    dataset_args={'dataset_id':str(data.resolve()),'subset_list':spec['subsets'],'few_shot_num':0}
    if args.benchmark=='live_code_bench':dataset_args['extra_params']={'start_date':spec['start_date'],'end_date':spec['end_date'],'timeout':6,'debug':False}
    config=TaskConfig(model=args.model,model_id=args.model,eval_type='openai_api',api_url=base+'/v1',api_key='EMPTY',
        model_args=protocol.get('model_args',{}),datasets=[args.benchmark],dataset_args={args.benchmark:dataset_args},generation_config=generation,
        eval_batch_size=args.eval_batch_size,seed=protocol['seed'],work_dir=str(work),use_cache=str(work),ignore_errors=False)
    provenance={'key':key,'execution':execution,'scoring':scoring,'protocol':protocol,'phase':args.phase,'thinking':args.thinking,'metadata':metadata,
        'selection':manifest,'code':code,'commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()}
    (work/'provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2)+'\n')
    started=time.time();result=run_task(config)
    from .audit import audit
    audit(work, manifest['count'])
    if sandbox.FAILURES:raise RuntimeError('Sandbox infrastructure errors: '+repr(sandbox.FAILURES))
    (work/'run_summary.json').write_text(json.dumps({'wall_seconds':time.time()-started,'result':result},default=str,indent=2)+'\n')
    print('EVAL_WORK_DIR='+str(work),flush=True)


if __name__=='__main__':main()

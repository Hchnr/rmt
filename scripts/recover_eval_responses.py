"""Resume frozen EvalScope runs while replaying durable orphaned API responses.

A failed concurrent prediction loop can leave successful raw responses without
prediction rows. Reuse their exact choices instead of sampling those prompts again.
The model, source, generation, dataset and runtime identities must still match.
"""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys
import threading
import requests
from transformers import AutoTokenizer
from openai.types.chat import ChatCompletion
from evalscope.models.openai_compatible import OpenAICompatibleAPI
from evalscope.models.utils.openai import openai_chat_messages, model_output_from_openai
from rmt.evaluation.protocol import request_seed
from rmt.evaluation.run import model_fingerprint, file_hash

class DurableResponses:
    def __init__(self, work):
        self.work=Path(work)
        self.provenance=json.loads((self.work/'provenance.json').read_text())
        self.tokenizer=AutoTokenizer.from_pretrained(self.provenance['metadata']['model'],local_files_only=True)
        self.seed=self.provenance['protocol']['seed']
        self.cache={};self.hits=[];self.lock=threading.Lock()
        for row in map(json.loads,(self.work/'responses.jsonl').read_text().splitlines()):
            meta=row['rmt_metadata'];key=(meta['prompt_sha256'],meta['seed'])
            if key in self.cache and self.cache[key]['choices']!=row['choices']:
                raise ValueError('Conflicting durable responses cannot be silently recovered')
            self.cache[key]=row

    def lookup(self, messages, seed):
        rendered=self.tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
        key=(hashlib.sha256(rendered.encode()).hexdigest(),seed)
        return key,self.cache.get(key)

    def install(self):
        original=OpenAICompatibleAPI.generate
        recovery=self
        def generate(api,input,tools,tool_choice,config):
            seed=request_seed(input,recovery.seed)
            if config.seed!=seed:raise ValueError('Request-specific seed changed')
            key,row=recovery.lookup(openai_chat_messages(input),seed)
            if row is None:return original(api,input,tools,tool_choice,config)
            if tools:raise ValueError('Durable text-only recovery does not accept tools')
            completion=ChatCompletion.model_validate(row)
            result=model_output_from_openai(completion,api.chat_choices_from_completion(completion,tools))
            with recovery.lock:recovery.hits.append({'prompt_sha256':key[0],'seed':key[1]})
            # Do not append the same response again; it is already durable.
            return result
        OpenAICompatibleAPI.generate=generate


def main():
    p=argparse.ArgumentParser();p.add_argument('--work',required=True);p.add_argument('--report',required=True)
    a,remaining=p.parse_known_args()
    if remaining and remaining[0]=='--':remaining=remaining[1:]
    def value(flag):return remaining[remaining.index(flag)+1]
    recovery=DurableResponses(a.work);original=recovery.provenance
    if '--thinking' in remaining:raise ValueError('This recovery only covers frozen non-thinking runs')
    profile=json.loads(Path(value('--config')).read_text())
    assert profile==original['protocol'],'Protocol changed'
    assert int(value('--eval-batch-size'))==original['execution']['eval_batch_size']
    assert value('--phase')==original['phase'] and value('--model')=='rmt'
    assert value('--benchmark')==recovery.work.parent.name
    manifest=Path(value('--data-root'))/value('--phase')/value('--benchmark')/'manifest.json'
    assert json.loads(manifest.read_text())==original['selection'],'Dataset selection changed'
    for name,sha in original['code'].items():assert file_hash(Path(name))==sha,name
    session=requests.Session();session.trust_env=False
    metadata=session.get('http://127.0.0.1:'+value('--port')+'/metadata',timeout=10).json()
    metadata['weights']=model_fingerprint(metadata['model'])
    assert metadata==original['metadata'],'Model/runtime/topology changed'
    before=(recovery.work/'responses.jsonl').read_bytes()
    report={'status':'running','work':a.work,'original_key':original['key'],'durable_unique_responses':len(recovery.cache),
            'original_responses_sha256':hashlib.sha256(before).hexdigest(),'source_runtime_protocol_equal':True}
    recovery.install()
    try:
        sys.argv=['rmt.evaluation.run',*remaining]
        runpy.run_module('rmt.evaluation.run',run_name='__main__')
        report['status']='passed'
    except BaseException:
        report['status']='failed';raise
    finally:
        after=(recovery.work/'responses.jsonl').read_bytes()
        report.update(replayed_responses=recovery.hits,original_response_prefix_unchanged=after.startswith(before),
                      response_lines_after=len(after.splitlines()))
        Path(a.report).write_text(json.dumps(report,indent=2)+'\n')
    assert report['original_response_prefix_unchanged']

if __name__=='__main__':main()

"""Verify real saved completions replay exactly with network access forbidden."""
import argparse
import hashlib
import json
from pathlib import Path
from evalscope.api.messages import ChatMessageUser
from evalscope.api.model import GenerateConfig
from evalscope.models.openai_compatible import OpenAICompatibleAPI
from rmt.evaluation.protocol import request_seed
from recover_eval_responses import DurableResponses

p=argparse.ArgumentParser();p.add_argument('--work',required=True);p.add_argument('--output',required=True);a=p.parse_args()
recovery=DurableResponses(a.work);response_path=Path(a.work)/'responses.jsonl'
before=hashlib.sha256(response_path.read_bytes()).hexdigest()
api=OpenAICompatibleAPI(model_name='rmt',base_url='http://127.0.0.1:1/v1',api_key='EMPTY')
def forbidden(**kwargs):raise RuntimeError('network disabled for recovery verification')
api.client.chat.completions.create=forbidden
recovery.install();count=0
for path in (Path(a.work)/'reviews').rglob('*.jsonl'):
    for row in map(json.loads,path.read_text().splitlines()):
        assert row['input'].startswith('**User**: ')
        text=row['input'].removeprefix('**User**: ')
        messages=[ChatMessageUser(content=text)]
        seed=request_seed(messages,recovery.seed)
        key,raw=recovery.lookup([{'role':'user','content':text}],seed)
        assert raw is not None,'Renderer or request identity mismatch'
        out=api.generate(messages,[],None,GenerateConfig(seed=seed))
        assert out.choices[0].message.content==raw['choices'][0]['message']['content']
        assert out.usage.output_tokens==raw['usage']['completion_tokens']
        count+=1
assert count>0 and len(recovery.hits)==count
try:
    api.generate(messages,[],None,GenerateConfig(seed=seed+1))
except ValueError as error:assert 'seed changed' in str(error)
else:raise AssertionError('Changed seed must fail')
unknown=[ChatMessageUser(content='unseen recovery control 123xyz')]
try:
    api.generate(unknown,[],None,GenerateConfig(seed=request_seed(unknown,recovery.seed)))
except RuntimeError as error:assert str(error)=='network disabled for recovery verification'
else:raise AssertionError('Unknown prompts must use the original API path')
assert hashlib.sha256(response_path.read_bytes()).hexdigest()==before
result={'status':'passed','records':count,'checks':['real saved text/usage preserved','no network calls for cached prompts','changed seed rejected','unknown prompt reaches original API path','raw response file unchanged']}
Path(a.output).write_text(json.dumps(result,indent=2)+'\n');print(result)

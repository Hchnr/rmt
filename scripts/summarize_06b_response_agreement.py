"""Measure answer agreement for an already audited native/untrained comparison."""
import argparse
import json
from pathlib import Path

parser=argparse.ArgumentParser()
parser.add_argument('--comparison',required=True)
parser.add_argument('--output',required=True)
args=parser.parse_args()
comparison=json.loads(Path(args.comparison).read_text())
results=[]

def responses(work):
    result={}
    for line in (Path(work)/'responses.jsonl').read_text().split('\n'):
        if not line.strip():continue
        row=json.loads(line);meta=row['rmt_metadata']
        key=(meta['prompt_sha256'],meta['seed'])
        if key in result:assert result[key]['choices']==row['choices']
        result[key]=row
    return result

for row in comparison['rows']:
    assert row['native_audit']['status']==row['candidate_audit']['status']=='passed'
    left=responses(row['native_work']);right=responses(row['candidate_work'])
    assert left.keys()==right.keys() and len(left)==row['n']
    same_text=same_finish=same_tokens=0
    changed=[]
    for key in sorted(left):
        a,b=left[key],right[key];x,y=a['choices'][0],b['choices'][0]
        equal=x['message']['content']==y['message']['content']
        same_text+=equal
        same_finish+=x['finish_reason']==y['finish_reason']
        same_tokens+=a['usage']['completion_tokens']==b['usage']['completion_tokens']
        if not equal:changed.append({'prompt_sha256':key[0],'seed':key[1],
            'native_completion_tokens':a['usage']['completion_tokens'],
            'rmt_completion_tokens':b['usage']['completion_tokens']})
    results.append({'benchmark':row['benchmark'],'n':row['n'],'same_answer_text':same_text,
        'same_finish_reason':same_finish,'same_completion_token_count':same_tokens,
        'changed_answers':changed,'native_work':row['native_work'],'rmt_work':row['candidate_work']})
Path(args.output).write_text(json.dumps({'rows':results,
    'scope':'Same frozen sampling protocol and request seeds. Measures observed answer agreement, not logits or token-ID bitwise equality; batch scheduling and backend numerics can affect sampled answers. Equal aggregate scores do not imply identical predictions.'},indent=2)+'\n')
print(json.dumps([{k:v for k,v in row.items() if k!='changed_answers'} for row in results],indent=2))

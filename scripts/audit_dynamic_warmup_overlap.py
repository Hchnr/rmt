"""Check new held-out prompts against all user turns in earlier warmup data."""
import hashlib
import json
from pathlib import Path
import unicodedata

def key(text):
    return ''.join(c for c in unicodedata.normalize('NFKC',text).casefold() if c.isalnum())

root=Path('artifacts/v0.0.4_dynamic_recurr/corpus')
old=Path('artifacts/v0.0.4/corpus/train_raw.jsonl')
retained={x['id'] for x in map(json.loads,Path('artifacts/v0.0.4/corpus/train.jsonl').read_text().splitlines())}
exposure=json.loads(Path('reports/v0.0.4_dynamic_recurr/warmup_training_exposure.json').read_text())['rows'][0]
warmup=json.loads(Path('reports/v0.0.4_dynamic_recurr/warmup.json').read_text())
assert exposure['data_sha256']==warmup['identity']['data']['train']
used=set(exposure['source_chunk_visits'])
assert used<=retained
users={}
for row in map(json.loads,old.read_text().splitlines()):
    if row['prompt_id'] not in retained:continue
    for index,message in enumerate(row['messages']):
        if message['role']=='user':
            users.setdefault(key(message['content']),[]).append({'id':row['prompt_id'],'turn':index,'used_document':row['prompt_id'] in used})
rows=[]
for split in ['halt_calibration','dev','test']:
    matches=[]
    for row in map(json.loads,(root/(split+'.jsonl')).read_text().splitlines()):
        prompt=key(row['messages'][0]['content'])
        for prior,origins in users.items():
            exact=prompt==prior
            contained=min(len(prompt),len(prior))>=80 and (prompt in prior or prior in prompt)
            if exact or contained:
                matches.append({'new_id':row['id'],'kind':'normalized_exact' if exact else 'normalized_containment','warmup':origins})
    rows.append({'split':split,'matches':matches,'used_document_matches':sum(any(x['used_document'] for x in match['warmup']) for match in matches)})
report={'status':'audited','warmup_raw_sha256':hashlib.sha256(old.read_bytes()).hexdigest(),
        'warmup_retained_documents':len(retained),'warmup_used_documents':len(used),'rows':rows,
        'scope':'Post-hoc deterministic normalized text/containment audit across all user turns; no semantic paraphrase guarantee. Does not alter frozen splits or scores; used_document can include a chunk without its original prompt.'}
Path('reports/v0.0.4_dynamic_recurr/warmup_overlap_audit.json').write_text(json.dumps(report,indent=2)+'\n')
for row in rows:print(row['split'],len(row['matches']),row['used_document_matches'])

import json
import torch
from rmt.training_data import load_packs,batch_at
from rmt.losses import shifted_targets
from rmt.train import prior_at


def test_packed_response_masks_and_chunk_boundaries(tmp_path):
    path=tmp_path/'tokens.jsonl'
    rows=[{'input_ids':[5,6,7,8,9,10],'labels':[-100,-100,7,8,9,10]},
          {'input_ids':[11,12,13],'labels':[-100,12,13]}]
    path.write_text(''.join(json.dumps(x)+'\n' for x in rows))
    packs=load_packs(path,4,0,shuffle=False)
    targets=[shifted_targets(x['labels'],x['attention_mask'],x['segment_ids']) for x in packs]
    supervised=[x for row in targets for x in row.flatten().tolist() if x!=-100]
    assert supervised==[7,8,10,12,13]  # 9 is the first token of a fresh chunk.
    for x in packs:
        assert torch.all(x['labels'][x['attention_mask']==0]==-100)
    assert torch.equal(batch_at(packs,1,0,2,'cpu')['input_ids'],packs[2]['input_ids'])


def test_prior_warmup_and_decay():
    cfg={'steps':100,'prior_start':4,'prior_end':2,'prior_warmup_steps':10,'prior_decay_steps':40}
    assert [prior_at(cfg,i) for i in [0,10,30,50,100]]==[4,4,3,2,2]


def test_unicode_line_separator_is_not_jsonl_record_boundary(tmp_path):
    path=tmp_path/'tokens.jsonl'
    row={'note':'first\u2028second','input_ids':[3,4,5],'labels':[-100,4,5]}
    path.write_text(json.dumps(row,ensure_ascii=False)+'\n')
    assert len(load_packs(path,4,0))==1

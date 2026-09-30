import json
import pytest
from rmt.evaluation.audit import audit


def test_audit_rejects_silently_missing_and_empty_scores(tmp_path):
    folder=tmp_path/'reviews'/'rmt';folder.mkdir(parents=True)
    row={'sample_score':{'sample_id':1,'score':{'value':{'acc':1},'metadata':{}}}}
    file=folder/'math_500.jsonl';file.write_text(json.dumps(row)+'\n')
    with pytest.raises(RuntimeError,match='Incomplete'):audit(tmp_path,2)
    row['sample_score']['score']['value']={};file.write_text(json.dumps(row)+'\n')
    with pytest.raises(RuntimeError,match='Invalid scorer'):audit(tmp_path,1)
    row['sample_score']['score']['value']={'acc':0};file.write_text((json.dumps(row)+'\n')*2)
    with pytest.raises(RuntimeError,match='Duplicate'):audit(tmp_path,2)


def test_seed_ignores_generated_message_ids():
    from rmt.evaluation.protocol import request_seed
    a=[{'role':'user','content':'1+1?','id':'rmt-random-id'}]
    b=[{'role':'user','content':'1+1?','id':'qwen-other-id'}]
    assert request_seed(a,17)==request_seed(b,17)
    assert request_seed(a,17)!=request_seed(a,18)
    assert request_seed(a,17)!=request_seed([{'role':'user','content':'2+2?'}],17)


def test_generation_policy_rejects_missing_native_eos():
    import pytest
    from rmt.evaluation.run import validate_generation_termination
    protocol={'expected_eos_token_ids':[151643,151645]}
    validate_generation_termination({'generation_termination':{'eos_token_ids':[151645,151643]}},protocol)
    with pytest.raises(ValueError,match='EOS policy mismatch'):
        validate_generation_termination({'generation_termination':{'eos_token_ids':[151645]}},protocol)
    with pytest.raises(ValueError,match='EOS policy mismatch'):
        validate_generation_termination({},protocol)

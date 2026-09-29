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

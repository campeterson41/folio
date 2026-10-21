import hashlib
import json
import pytest
from voicelab.f_artifacts import verify


def test_selected_checkpoint_is_verified_before_use(tmp_path):
    model=tmp_path/'model.pt';model.write_bytes(b'test checkpoint')
    lock=tmp_path/'lock.json'
    data={'artifacts':{'model.pt':{'path':'model.pt','sha256':hashlib.sha256(model.read_bytes()).hexdigest()}},
          'source':{'revision':'fixed-source'},'model_revision':'fixed-model'}
    lock.write_text(json.dumps(data))
    result=verify(tmp_path,lock,verify_source=False)
    assert result['model_revision']=='fixed-model'
    model.write_bytes(b'another checkpoint')
    with pytest.raises(ValueError,match='differs from the selected checkpoint'):
        verify(tmp_path,lock,verify_source=False)

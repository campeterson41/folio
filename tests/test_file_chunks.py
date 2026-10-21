import numpy as np
import soundfile as sf
import pytest
from voicelab.chunks import restore_chunks, restore_file


@pytest.mark.parametrize('seconds',[1.2,3.,6.01,10.3])
def test_file_restoration_matches_global_grid_and_overlap(seconds,tmp_path):
    rate=48000
    x=np.random.default_rng(11).normal(0,.05,round(seconds*rate))
    source=tmp_path/'in.wav';target=tmp_path/'out.wav'
    sf.write(source,x,rate,subtype='DOUBLE')
    seen=[]
    def infer(block):
        seen.append(len(block))
        # Depends on the exact context window, not just each sample.
        return block*.8 + np.mean(block)*.2
    expected=restore_chunks(x,rate,infer,crossfade_seconds=.08)
    stats=restore_file(source,target,infer)
    actual,_=sf.read(target)
    np.testing.assert_array_equal(actual,expected)
    assert stats['samples']==len(x)
    assert max(seen)<=4*rate


def test_file_restoration_rejects_missing_model_samples(tmp_path):
    source=tmp_path/'in.wav'
    sf.write(source,np.ones(48000)*.1,48000)
    with pytest.raises(ValueError):
        restore_file(source,tmp_path/'out.wav',lambda x:x[:1])

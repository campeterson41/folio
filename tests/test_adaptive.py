import numpy as np
import soundfile as sf
import pytest
from voicelab.adaptive import analyze,controller,protect

def fixture(path,offset=0):
    sr=48000;t=np.arange(sr*9)/sr
    pitch=np.where((t>3+offset)&(t<4+offset),230,110)
    phase=np.cumsum(2*np.pi*pitch/sr)
    env=((t>.5)&(t<8.5))*.04
    x=env*(np.sin(phase)+.3*np.sin(2*phase))
    sf.write(path,x,sr,subtype='PCM_24')
    return x

@pytest.mark.parametrize('offset',[0,2])
def test_register_routing_moves_with_performance_not_timestamp(tmp_path,offset):
    path=tmp_path/'source.wav';fixture(path,offset)
    features=analyze(path);plan=controller(features)
    assert 100<plan['baseline_pitch_hz']<120
    assert plan['selected'][round((3.5+offset)*50)]
    assert not plan['selected'][:20].any()

def test_noise_and_silence_are_not_detected_as_high_register(tmp_path):
    path=tmp_path/'noise.wav';sf.write(path,np.random.default_rng(5).normal(0,1e-6,48000*3),48000)
    assert not controller(analyze(path))['regions']

def test_protection_preserves_timing_inputs_and_regular_voice(tmp_path):
    source=tmp_path/'source.wav';x=fixture(source)
    # Pause separates regular narration from character delivery.
    x[round(2.8*48000):round(3*48000)]=0
    x[round(4*48000):round(4.2*48000)]=0
    sf.write(source,x,48000,subtype='PCM_24')
    wet=tmp_path/'wet.wav';dry=tmp_path/'gentle.wav'
    sf.write(wet,x*.5,48000,subtype='PCM_24');sf.write(dry,x,48000,subtype='PCM_24')
    before=[p.read_bytes() for p in (source,wet,dry)]
    target=tmp_path/'out.wav';info=protect(source,wet,dry,target,chunk_seconds=2.)
    y,_=sf.read(target);z,_=sf.read(wet)
    assert len(y)==len(x)
    assert any(a<3.5<b for a,b in info['regions_seconds'])
    np.testing.assert_array_equal(y[:48000*2],z[:48000*2])
    assert before==[p.read_bytes() for p in (source,wet,dry)]
    with pytest.raises(FileExistsError):protect(source,wet,dry,target)

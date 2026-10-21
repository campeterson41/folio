import numpy as np
import pytest
import soundfile as sf
from scipy import signal
from voicelab.finish_tone import bass_sos,bass_file

def test_subtle_shelf_lifts_bass_and_leaves_presence_alone():
    _,h=signal.sosfreqz(bass_sos(.5),worN=np.array([30.,160.,1000.,5000.]),fs=48000)
    db=20*np.log10(abs(h))
    assert .49<db[0]<.51
    assert abs(db[1]-.25)<.001
    assert max(abs(db[2:]))<.001

def test_disk_boundaries_do_not_reset_eq_state(tmp_path):
    rng=np.random.default_rng(1);x=rng.normal(0,.01,48000*3+177)
    source=tmp_path/'source.wav';sf.write(source,x,48000,subtype='PCM_24')
    before=source.read_bytes();a=tmp_path/'a.wav';b=tmp_path/'b.wav'
    info=bass_file(source,a,.5,chunk_seconds=.37);bass_file(source,b,.5,chunk_seconds=2.)
    assert a.read_bytes()==b.read_bytes()
    assert info['samples']==len(x) and source.read_bytes()==before
    with pytest.raises(FileExistsError):bass_file(source,a,.5)
    with pytest.raises(ValueError):bass_file(source,source,.5)

def test_zero_is_exact_bypass_and_excess_gain_is_rejected(tmp_path):
    source=tmp_path/'source.wav';target=tmp_path/'out.wav'
    sf.write(source,np.zeros(4800),48000,subtype='PCM_24')
    bass_file(source,target,0);assert source.read_bytes()==target.read_bytes()
    for gain in (float('nan'),4.):
        with pytest.raises(ValueError):bass_sos(gain)

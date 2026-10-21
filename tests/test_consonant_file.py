import io
import numpy as np
import pytest
import soundfile as sf
from scipy import signal
from voicelab.consonants import preserve_fullband
from voicelab.consonant_file import preserve_file


@pytest.mark.parametrize('start',[4.95,9.95])
def test_global_preservation_crosses_disk_boundaries_without_retiming(tmp_path,monkeypatch,start):
    rate=48000;n=rate*12;t=np.arange(n)/rate
    rng=np.random.default_rng(24)
    body=.04*np.sin(2*np.pi*700*t)*((t>.5)&(t<11.5))
    hiss=signal.sosfilt(signal.butter(3,[5000,13000],fs=rate,btype='bandpass',output='sos'),rng.normal(0,.1,n))
    burst=hiss*((t>start)&(t<start+.16))
    source=tmp_path/'source.wav';wet=tmp_path/'wet.wav';out=tmp_path/'out.wav'
    sf.write(source,body+burst,rate,subtype='DOUBLE')
    sf.write(wet,body+burst*.01,rate,subtype='PCM_24')
    dry,_=sf.read(source);enhanced,_=sf.read(wet)
    expected,expected_info=preserve_fullband(dry,enhanced)
    buf=io.BytesIO();sf.write(buf,expected,rate,subtype='PCM_24',format='WAV')
    buf.seek(0);expected,_=sf.read(buf)
    reads=[];original=sf.SoundFile.read
    def read(self,frames=-1,*args,**kwargs):
        reads.append(frames)
        return original(self,frames,*args,**kwargs)
    with monkeypatch.context() as m:
        m.setattr(sf.SoundFile,'read',read)
        info=preserve_file(source,wet,out,chunk_seconds=5)
    actual,_=sf.read(out)
    assert info['regions_seconds']==expected_info['regions_seconds']
    assert any(a<=start+.08<=b for a,b in info['regions_seconds'])
    assert all(0<=r<=rate*13 for r in reads)
    assert len(actual)==len(dry)
    assert np.max(abs(actual-expected))<=2/8388608


def test_file_preservation_silence_and_input_protection(tmp_path):
    source=tmp_path/'source.wav';wet=tmp_path/'wet.wav';out=tmp_path/'out.wav'
    sf.write(source,np.zeros(4800),48000,subtype='PCM_24')
    sf.write(wet,np.zeros(4800),48000,subtype='PCM_24')
    info=preserve_file(source,wet,out)
    assert info['regions_seconds']==[]
    assert out.read_bytes()==wet.read_bytes()
    with pytest.raises(ValueError):preserve_file(source,wet,source)
    with pytest.raises(FileExistsError):preserve_file(source,wet,out)

import io
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from voicelab import dsp
from voicelab.f_master_file import master_file
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from apply_f_master import apply


def test_file_master_preserves_f_across_boundaries_and_bounds_reads(tmp_path,monkeypatch):
    rate=48000;t=np.arange(rate*17)/rate
    x=.04*(.7+.3*np.sin(t*.7))*np.sin(2*np.pi*220*t)
    x+=.01*np.sin(2*np.pi*2400*t)
    x[5*rate]=.65
    source=tmp_path/'in.wav';target=tmp_path/'out.wav'
    sf.write(source,x,rate,subtype='DOUBLE')
    preset={'tone_profile':{'frequencies_hz':[60,18000],'gains_db':[1.5,-1.5]},
            'mastering_settings':{'tone_strength':1,'level_strength':.4,'ratio':1.3,
                                  'cutoff':50,'deess_strength':0,'limiter':False},
            'preferred_final_eq':{'type':'peaking_biquad','hz':2800,'gain_db':-.25,'q':.7}}
    expected,_=apply(x,preset)
    buf=io.BytesIO();sf.write(buf,expected,rate,format='WAV',subtype='PCM_24')
    buf.seek(0);expected,_=sf.read(buf)
    reads=[]
    original=sf.SoundFile.read
    def read(self,frames=-1,*args,**kwargs):
        reads.append(frames)
        return original(self,frames,*args,**kwargs)
    with monkeypatch.context() as m:
        m.setattr(sf.SoundFile,'read',read)
        info=master_file(source,target,preset,chunk_seconds=5)
    assert all(0<=n<=rate*13 for n in reads)
    actual,_=sf.read(target)
    assert info['samples']==len(x)==len(actual)
    assert np.max(abs(actual-expected))<=2/8388608

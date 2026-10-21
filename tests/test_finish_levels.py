import numpy as np
import soundfile as sf
from voicelab.finish_levels import finish_file


def test_a_loud_outlier_does_not_set_the_whole_chapter_level(tmp_path):
    sr=48000;t=np.arange(sr*12)/sr
    x=.012*np.sin(2*np.pi*220*t)
    x[sr*5]=.6
    source=tmp_path/'in.wav';target=tmp_path/'out.wav'
    sf.write(source,x,sr,subtype='PCM_24')
    before=source.read_bytes()
    info=finish_file(source,target,chunk_seconds=3)
    y,_=sf.read(target)
    assert len(y)==len(x)
    assert source.read_bytes()==before
    assert abs(info['final_rms_dbfs']+20)<.1
    assert info['final_true_peak_dbfs']<=-3.999
    assert max(info['limiter_max_reduction_each_pass_db'])>10
    # Regular speech is scaled uniformly, including across disk-block joins.
    mask=((t<4.9)|(t>6))&(abs(x)>.001)
    ratios=y[mask]/x[mask]
    assert np.percentile(ratios,99)-np.percentile(ratios,1)<.03


def test_level_finishing_preserves_silence(tmp_path):
    source=tmp_path/'in.wav';target=tmp_path/'out.wav'
    sf.write(source,np.zeros(48000),48000,subtype='PCM_24')
    info=finish_file(source,target)
    assert info['bypassed']=='silence or near-silence'
    assert source.read_bytes()==target.read_bytes()


def test_finishing_honors_its_boost_cap(tmp_path):
    t=np.arange(48000)/48000
    source=tmp_path/'in.wav';target=tmp_path/'out.wav'
    sf.write(source,.004*np.sin(2*np.pi*220*t),48000,subtype='PCM_24')
    info=finish_file(source,target,maximum_boost_db=6)
    assert info['total_uniform_boost_db']<=6.00001
    assert info['final_rms_dbfs'] < -40

"""Actual EBU meter + DSP regressions, without neural inference."""
import shutil
import numpy as np
import pytest
import soundfile as sf
from voicelab import studio_mastering as M, master_me as MM

def engine_available():
    try: MM.engine(); return True
    except ValueError: return False

pytestmark = pytest.mark.skipif(shutil.which('ffmpeg') is None or not engine_available(), reason='requires FFmpeg and installed master_me')


def voice():
    t=np.arange(8*48000)/48000
    envelope=(.3+.7*np.sin(2*np.pi*.8*t)**2)*((t%2)<1.4)
    return envelope*(np.sin(2*np.pi*173*t)+.15*np.sin(2*np.pi*519*t))


@pytest.mark.parametrize('db', [-28, -20, -5])
def test_quiet_and_loud_inputs_converge_without_clipping(tmp_path, db):
    x=voice()*10**(db/20)
    src=tmp_path/'input.wav';out=tmp_path/'output.wav'
    sf.write(src,x,48000,subtype='DOUBLE')
    details=M.master(src,out)
    assert details['target_reached']
    assert abs(details['output']['integrated_lufs']+26)<.15
    assert details['output']['true_peak_db']<=-3
    assert details['samples']==len(x)
    assert sf.info(out).subtype=='PCM_24'
    if db == -5: assert details['uniform_gain_db']<0
    if db == -28: assert details['uniform_gain_db']>0


def test_peaky_audio_gets_conditional_peak_control(tmp_path):
    x=voice()*.025
    x[3*48000:3*48000+480]=.95*np.sin(2*np.pi*400*np.arange(480)/48000)
    src=tmp_path/'input.wav';sf.write(src,x,48000,subtype='DOUBLE')
    details=M.master(src,tmp_path/'out.wav')
    y=sf.read(tmp_path/'out.wav')[0]
    before=np.max(abs(x))/np.sqrt(np.mean(x*x))
    after=np.max(abs(y))/np.sqrt(np.mean(y*y))
    assert 20*np.log10(before/after)>2
    assert details['engine']['engine']=='master_me'
    assert details['output']['true_peak_db']<=-3
    assert details['samples']==len(x)


def test_quiet_material_has_bounded_gain_and_silence_is_rejected(tmp_path):
    src=tmp_path/'input.wav'
    sf.write(src,voice()*1e-3,48000,subtype='DOUBLE')
    details=M.master(src,tmp_path/'out.wav')
    assert details['uniform_gain_db']<=12
    assert not details['target_reached']
    sf.write(src,np.zeros(48000*5),48000,subtype='DOUBLE')
    with pytest.raises(ValueError,match='measurable'):
        M.master(src,tmp_path/'silence.wav')
    assert not (tmp_path/'silence.wav').exists()


def test_master_never_overwrites(tmp_path):
    out=tmp_path/'existing.wav';out.write_bytes(b'keep')
    with pytest.raises(FileExistsError):M.master(tmp_path/'missing.wav',out)
    assert out.read_bytes()==b'keep'


def test_varying_delivery_gets_adaptive_dynamics(tmp_path):
    x=np.tile(voice(),4)*.07
    x[:len(x)//2]*=.16
    src=tmp_path/'uneven.wav';sf.write(src,x,48000,subtype='DOUBLE')
    M.master(src,tmp_path/'even.wav')
    y=sf.read(tmp_path/'even.wav')[0];mid=len(x)//2
    def difference(a):
        return 10*np.log10(np.mean(a[mid:]**2)/np.mean(a[:mid]**2))
    assert difference(y)<difference(x)-1
    assert len(x)==len(y)


def test_modified_engine_is_rejected_before_processing(tmp_path):
    import json
    plugin,manifest=MM.engine()
    target=tmp_path/'mastering';target.mkdir()
    shutil.copy2(plugin,target/plugin.name)
    (target/'manifest.json').write_text(json.dumps(manifest))
    with (target/plugin.name).open('ab') as f:f.write(b'corrupt')
    with pytest.raises(ValueError,match='identity mismatch'):
        MM.engine(tmp_path)

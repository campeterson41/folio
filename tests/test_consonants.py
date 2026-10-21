import numpy as np
from voicelab.consonants import preserve
from voicelab.consonants import preserve_fullband


def test_fullband_detector_follows_signal_at_different_positions():
    from scipy import signal
    rate = 48000
    rng = np.random.default_rng(8)
    n = rate*5
    t = np.arange(n)/rate
    body = .05*np.sin(2*np.pi*220*t)*((t>.4)&(t<4.6))
    hiss = signal.sosfilt(signal.butter(3,[5000,13000],fs=rate,btype='bandpass',output='sos'),rng.normal(0,.1,n))
    for start in [1.2, 3.3]:
        burst = hiss*((t>start)&(t<start+.18))
        dry = body+burst+rng.normal(0,1e-6,n)
        wet = body+burst*.02
        out, info = preserve_fullband(dry,wet,delay_samples=0)
        assert any(a <= start+.08 <= b for a,b in info['regions_seconds'])
        sl = slice(round((start+.05)*rate),round((start+.13)*rate))
        assert np.linalg.norm(out[sl]-dry[sl]) < np.linalg.norm(wet[sl]-dry[sl])*.5
        np.testing.assert_array_equal(out[:rate//3],wet[:rate//3])


def test_fullband_does_not_restore_constant_hiss_or_a_click():
    rate=48000;n=rate*3;t=np.arange(n)/rate
    rng=np.random.default_rng(2)
    body=.08*np.sin(2*np.pi*180*t)
    dry=body+rng.normal(0,.003,n)
    wet=body.copy()
    dry[rate]=1
    out,info=preserve_fullband(dry,wet,delay_samples=0)
    np.testing.assert_array_equal(out,wet)
    assert info['regions_seconds']==[]


def test_quiet_fricative_is_not_hidden_by_a_remote_loud_sound():
    from scipy import signal
    rate=48000;n=rate*8;t=np.arange(n)/rate
    rng=np.random.default_rng(19)
    body=.0005*np.sin(2*np.pi*700*t)*((t>1)&(t<7))
    loud=.8*np.sin(2*np.pi*700*t)*((t>6)&(t<6.1))
    hiss=signal.sosfilt(signal.butter(3,[5000,13000],fs=rate,btype='bandpass',output='sos'),rng.normal(0,.0004,n))
    burst=hiss*((t>3)&(t<3.15))
    source=body+loud+burst
    wet=body+loud+burst*.01
    _,old=preserve_fullband(source,wet,delay_samples=0)
    out,new=preserve_fullband(source,wet,delay_samples=0,voiced_fricatives=True)
    assert not any(a<=3.08<=b for a,b in old['regions_seconds'])
    assert any(a<=3.08<=b for a,b in new['regions_seconds'])
    assert np.linalg.norm(out[3*rate:round(3.15*rate)]-source[3*rate:round(3.15*rate)]) < np.linalg.norm(wet[3*rate:round(3.15*rate)]-source[3*rate:round(3.15*rate)])

def test_preserves_a_missing_fricative_without_changing_quiet():
    rate=48000;rng=np.random.default_rng(42);n=rate*4;t=np.arange(n)/rate
    body=.08*np.sin(2*np.pi*220*t)*((t>.5)&(t<3.5))
    noise=rng.normal(0,.000001,n)
    hiss=rng.normal(0,.03,n)*((t>1.5)&(t<1.8))
    source=body+noise+hiss;wet=body+noise+hiss*.03
    out,info=preserve(source,wet,strength=.25,delay_samples=0)
    assert info['active_seconds']>0
    sl=slice(int(1.55*rate),int(1.75*rate))
    assert np.linalg.norm(out[sl]-source[sl])<np.linalg.norm(wet[sl]-source[sl])
    np.testing.assert_array_equal(out[:rate//3],wet[:rate//3])
    np.testing.assert_array_equal(out[3*rate+rate*3//4:],wet[3*rate+rate*3//4:])

def test_bypass_silence_and_shape_validation():
    import pytest
    x=np.zeros(48000);out,info=preserve(x,x)
    np.testing.assert_array_equal(out,x)
    with pytest.raises(ValueError):preserve(x,x[:-1])
    with pytest.raises(ValueError):preserve(x,x,strength=2)

import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from scipy import signal

from voicelab import source_pipeline as p

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
import folio_source  # noqa: E402
from synthetic_speech import narration  # noqa: E402

RATE = p.RATE


def preset():
    return json.loads(p_path().read_text())


def p_path():
    return ROOT/'presets/source-narration-v1.json'


def rms_db(x):
    return 10*np.log10(np.mean(x**2)+1e-30)


@pytest.fixture(scope='module')
def rendered(tmp_path_factory):
    d = tmp_path_factory.mktemp('render')
    x, gate = narration()
    src = d/'take.wav'
    sf.write(src, x, RATE, subtype='PCM_24')
    before = src.read_bytes()
    out = d/'out.wav'
    opened = []
    real_open, real_socket = sf.SoundFile.__init__, socket.socket.connect

    def watch(self, file, *a, **k):
        opened.append(Path(file).name)
        return real_open(self, file, *a, **k)

    def no_network(*a, **k):
        raise AssertionError('network access')
    mp = pytest.MonkeyPatch()
    mp.setattr(sf.SoundFile, '__init__', watch)
    mp.setattr(socket.socket, 'connect', no_network)
    try:
        folio_source.main(['--input', str(src), '--output', str(out), '--quiet'])
    finally:
        mp.undo()
    assert src.read_bytes() == before
    y, rate = sf.read(out)
    return dict(src=src, out=out, x=sf.read(src)[0], y=y, rate=rate, gate=gate,
                opened=opened, receipt=json.loads(out.with_suffix('.json').read_text()))


def test_end_to_end_keeps_timing_and_is_safe(rendered):
    r = rendered
    x, y = r['x'], r['y']
    assert r['rate'] == RATE and len(y) == len(x)
    assert sf.info(r['out']).subtype == 'PCM_24'
    assert np.isfinite(y).all() and np.max(abs(y)) < 10**(-1/20)
    # Only the input and private intermediates are opened; no reference audio.
    allowed = {'take.wav', 'preserved-toned.wav', 'preserved.wav', 'quieter.wav',
               'controlled.wav', 'final.wav'}
    assert set(r['opened']) <= allowed and 'take.wav' in r['opened']
    rec = r['receipt']
    assert rec['samples'] == len(x) and rec['output_sha256'] == folio_source.sha(r['out'])
    assert rec['source_sha256'] == folio_source.sha(r['src'])
    assert rec['neural_models_used'] is False and rec['reference_audio_read'] is False
    assert rec['settings']['maximum_tail_attenuation_db'] == 6
    assert rec['settings']['bass_shelf_db'] == 3
    assert len(rec['implementation_digest']) == 64
    assert sorted(q.name for q in r['out'].parent.iterdir()) == ['out.json', 'out.wav', 'take.wav']


def test_end_to_end_reduces_noise_between_phrases(rendered):
    x, y, gate = rendered['x'], rendered['y'], rendered['gate']
    pause = slice(RATE//4, int(1.75*RATE))
    assert not gate[:2*RATE].any()
    speech = gate > .5
    before = rms_db(x[speech])-rms_db(x[pause])
    after = rms_db(y[speech])-rms_db(y[pause])
    assert after > before+15
    # Speech itself is not removed: within a broad band of the fixed preset gain.
    assert -6 < rms_db(y[speech])-rms_db(x[speech]) < 10


def test_no_duration_shift_in_output(rendered):
    x, y = rendered['x'], rendered['y']
    a, b = x[10*RATE:12*RATE], y[10*RATE:12*RATE]
    c = signal.correlate(b, a, 'full', method='fft')
    lag = int(np.argmax(c[len(a)-1-200:len(a)+200]))-200
    assert lag == 0


def test_output_collision_and_input_rejection(tmp_path, rendered):
    src = rendered['src']
    for bad in (rendered['out'], rendered['out'].with_suffix('.json')):
        target = tmp_path/'x.wav'
        if bad.suffix == '.json':
            target.with_suffix('.json').write_text('{}')
        else:
            target = bad
        with pytest.raises(FileExistsError):
            folio_source.main(['--input', str(src), '--output', str(target), '--quiet'])
    with pytest.raises(SystemExit):
        folio_source.main(['--input', str(src), '--output', str(src), '--quiet'])
    x, _ = narration(15, seed=3, lead=.8)
    for name, data, rate, subtype in (
            ('rate.wav', x, 44100, 'PCM_24'),
            ('stereo.wav', np.c_[x, x], RATE, 'PCM_24'),
            ('short.wav', x[:3*RATE], RATE, 'PCM_24'),
            ('nan.wav', np.r_[x[:-1], np.nan], RATE, 'FLOAT')):
        f = tmp_path/name
        sf.write(f, data, rate, subtype=subtype)
        with pytest.raises(ValueError):
            folio_source.main(['--input', str(f), '--output', str(tmp_path/('o'+name)),
                               '--quiet'])
        assert not (tmp_path/('o'+name)).exists()


def test_continuous_noise_without_pauses_fails_clearly(tmp_path):
    rng = np.random.default_rng(4)
    t = np.arange(10*RATE)/RATE
    x = .1*np.sin(2*np.pi*180*t)*(1+.3*np.sin(2*np.pi*4*t))+rng.normal(0, .05, len(t))
    f = tmp_path/'busy.wav'
    sf.write(f, x, RATE, subtype='PCM_24')
    with pytest.raises(ValueError, match='quiet|noise reference|speech'):
        folio_source.main(['--input', str(f), '--output', str(tmp_path/'o.wav'), '--quiet'])
    assert not (tmp_path/'o.wav').exists()


def test_interruption_removes_temporary_files(tmp_path, monkeypatch):
    x, _ = narration(15, seed=5, lead=.8)
    f = tmp_path/'take.wav'
    sf.write(f, x, RATE, subtype='PCM_24')
    real = p.stage
    calls = []

    def interrupted(*a, **k):
        calls.append(a[3])
        if len(calls) == 3:
            raise KeyboardInterrupt
        return real(*a, **k)
    monkeypatch.setattr(p, 'stage', interrupted)
    with pytest.raises(KeyboardInterrupt):
        folio_source.main(['--input', str(f), '--output', str(tmp_path/'o.wav'), '--quiet'])
    assert sorted(q.name for q in tmp_path.iterdir()) == ['take.wav']


def test_loud_peaks_are_limited_and_reported(tmp_path):
    x, _ = narration(15, seed=9, speech_peak=.5, lead=.8)
    x[7*RATE:7*RATE+2400] += .45*np.sin(2*np.pi*300*np.arange(2400)/RATE)
    x = np.clip(x, -.999, .999)
    f = tmp_path/'loud.wav'
    sf.write(f, x, RATE, subtype='PCM_24')
    out = tmp_path/'o.wav'
    folio_source.main(['--input', str(f), '--output', str(out), '--quiet'])
    y, _ = sf.read(out)
    rec = json.loads(out.with_suffix('.json').read_text())
    assert np.max(abs(y)) < .99 and rec['metrics']['estimated_true_peak_4x'] < .99
    assert rec['limiter_max_reduction_db']['quieter'] > 3


def test_cli_wrapper_runs_without_model_runtime(tmp_path):
    x, _ = narration(15, seed=11, lead=.8)
    f = tmp_path/'take.wav'
    sf.write(f, x, RATE, subtype='PCM_24')
    probe = ("import runpy,sys;sys.argv=sys.argv[1:];"
             "runpy.run_path(sys.argv[0],run_name='__main__')")
    script = ROOT/'tools/folio_source.py'
    env = {**os.environ, 'PYTHONPATH': ''}
    done = subprocess.run([sys.executable, '-c', probe+";import sys as s;"
                           "assert not {'torch','onnxruntime','requests'}&set(s.modules)",
                           str(script), '--input', str(f), '--output', str(tmp_path/'o.wav'),
                           '--quiet'], env=env, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    help_text = subprocess.run([str(ROOT/'bin/folio-source'), '--help'],
                               env={**env, 'FOLIO_PYTHON': sys.executable},
                               capture_output=True, text=True)
    assert help_text.returncode == 0 and '48 kHz mono' in help_text.stdout


# Generic signal checks of individual stages.

def test_noise_reduction_attenuates_steady_noise_and_keeps_tone():
    rng = np.random.default_rng(21)
    n = 6*RATE
    noise = rng.normal(0, .003, n)
    t = np.arange(n)/RATE
    tone = .1*np.sin(2*np.pi*440*t)*(t > 3)
    _, _, z = signal.stft(noise[:RATE], RATE, nperseg=p.FFT, noverlap=p.FFT-p.STEP,
                          boundary=None)
    profile = np.median(abs(z)**2, axis=1)
    # A median noise profile underestimates mean noise power, so the 12 dB
    # branch removes roughly 5-6 dB of white noise and the 1.6x branch more.
    for fn, low, cap in ((lambda v: p.reduce_noise(v, profile, 12), 4, 12),
                         (lambda v: p.clean_quieter(v, profile), 11, 18)):
        y = fn(noise+tone)
        assert len(y) == n
        quiet = slice(RATE, 2*RATE)
        reduction = rms_db(noise[quiet])-rms_db(y[quiet])
        assert low < reduction < cap+.5
        loud = slice(4*RATE, 5*RATE)
        assert abs(rms_db(y[loud])-rms_db((noise+tone)[loud])) < 1


def test_level_controls_reduce_pauses_but_not_speech():
    level = np.full(1000, -70.)
    level[300:500] = -25
    level[300:310] = -10
    wet = level+3
    gains, info = p.controls({'level_dbfs': level}, {'level_dbfs': wet})
    assert gains.max() <= 1e-9 and gains.min() >= -11.5
    assert np.all(gains[320:480] > -2.6)
    assert np.all(gains[:200] < -8.9) and np.all(gains[650:] < -8.9)
    assert info['maximum_quiet_at_detected_speech_db'] < 1e-6
    with pytest.raises(ValueError):
        p.controls({'level_dbfs': np.full(100, -70.)}, {'level_dbfs': np.full(100, -70.)})


def test_shelf_increases_bass_relative_to_midrange_without_delay():
    t = np.arange(RATE*2)/RATE
    x = .03*np.sin(2*np.pi*90*t) + .03*np.sin(2*np.pi*1000*t)
    y = p.closer_warm(x, 0)
    keep = slice(RATE//2, -RATE//2)

    def component(v, hz):
        return abs(np.mean(v[keep]*np.exp(-2j*np.pi*hz*t[keep])))
    delta = 20*np.log10(component(y, 90)/component(x, 90))
    assert 2.5 < delta < 3.1
    assert abs(20*np.log10(component(y, 1000)/component(x, 1000))) < .05
    impulse = np.zeros(RATE)
    impulse[RATE//2] = 1
    assert np.argmax(p.closer_warm(impulse, 0)) == RATE//2


def test_mask_is_positive_bounded_and_reduces_decaying_tail():
    rng = np.random.default_rng(31)
    x = rng.normal(0, .02, RATE*3)
    # A sustained excitation followed by a decaying late field.
    x[RATE:] *= np.exp(-np.arange(RATE*2)/RATE/.09)
    _, _, z = signal.stft(x, RATE, nperseg=p.FFT, noverlap=p.FFT-p.STEP)
    mask = p.tail_mask(z, 6)
    assert mask.min() >= 10**(-6/20)-1e-12 and mask.max() <= 1
    y = p.closer(x, 6)
    assert len(y) == len(x) and np.isfinite(y).all()
    tail = slice(int(1.05*RATE), int(1.3*RATE))
    assert np.sqrt(np.mean(y[tail]**2)/np.mean(x[tail]**2)) < .88
    steady = slice(RATE//2, int(.9*RATE))
    assert np.sqrt(np.mean(y[steady]**2)/np.mean(x[steady]**2)) > .9


@pytest.mark.parametrize('fn', [lambda x: p.closer_warm(x, 6),
                                lambda x: p.clean_quieter(x, np.full(1025, 1e-9)),
                                lambda x: p.reduce_noise(x, np.full(1025, 1e-9), 12)])
def test_chunk_grid_continuity_and_silence(fn):
    rng = np.random.default_rng(35)
    x = rng.normal(0, .02, RATE*5-137)
    x[RATE:RATE*2] *= np.exp(-np.arange(RATE)/RATE/.08)
    cut = round(2.5*RATE/p.STEP)*p.STEP
    context = round(p.CONTEXT_SECONDS*RATE)
    full = fn(x)
    joined = np.r_[fn(x[:cut+context])[:cut], fn(x[cut-context:])[context:]]
    assert np.max(abs(full-joined)) < 1e-7
    assert np.max(abs(fn(np.zeros(RATE)))) == 0
    with pytest.raises(ValueError):
        p.closer(x, 7)


def test_chunk_constants_are_on_the_stft_grid():
    assert round(p.CHUNK_SECONDS*RATE) % p.STEP == 0
    assert round(p.CONTEXT_SECONDS*RATE) % p.STEP == 0
    assert abs(p.CHUNK_SECONDS-30) < .01 and abs(p.CONTEXT_SECONDS-2) < .01


def test_preset_validation():
    good = preset()
    assert p.validate_preset(good)[1] == good['gain_db']
    for bad in ({**good, 'gain_db': float('nan')},
                {**good, 'tone': {'frequencies_hz': [100, 50], 'gains_db': [0, 0]}},
                {**good, 'tone': {'frequencies_hz': [50, 100], 'gains_db': [0, 9]}}):
        with pytest.raises(ValueError):
            p.validate_preset(bad)


def test_busy_output_lock_is_refused_and_left_alone(tmp_path):
    import fcntl
    src = tmp_path/'take.wav'
    sf.write(src, narration(6, seed=1, lead=.8)[0], RATE, subtype='PCM_24')
    lock_path = tmp_path/'.o.wav.lock'
    with lock_path.open('a') as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        with pytest.raises(FileExistsError, match='already being processed'):
            folio_source.main(['--input', str(src), '--output', str(tmp_path/'o.wav'),
                               '--quiet'])
        assert lock_path.exists() and not (tmp_path/'o.wav').exists()


def test_interrupt_after_wav_link_removes_only_owned_output(tmp_path, rendered, monkeypatch):
    import shutil
    output=tmp_path/'complete.wav'
    receipt=output.with_suffix('.json')
    def fake_render(source, workdir, *args, **kwargs):
        final=Path(workdir)/'final.wav'
        shutil.copyfile(rendered['out'],final)
        return final, {'limiter_max_reduction_db':{'preserved':0,'quieter':0}}
    real_link=os.link
    def interrupted(source, destination, *args, **kwargs):
        real_link(source,destination,*args,**kwargs)
        if Path(destination)==output:
            raise KeyboardInterrupt
    monkeypatch.setattr(p,'render',fake_render)
    monkeypatch.setattr(os,'link',interrupted)
    with pytest.raises(KeyboardInterrupt):
        folio_source.main(['--input',str(rendered['src']),'--output',str(output),'--quiet'])
    assert not output.exists() and not receipt.exists()
    assert not list(tmp_path.glob('.folio-source-*'))


def test_receipt_race_keeps_other_writers_file(tmp_path, rendered, monkeypatch):
    import shutil
    output=tmp_path/'complete.wav'
    receipt=output.with_suffix('.json')
    def fake_render(source, workdir, *args, **kwargs):
        final=Path(workdir)/'final.wav'
        shutil.copyfile(rendered['out'],final)
        return final, {'limiter_max_reduction_db':{'preserved':0,'quieter':0}}
    real_link=os.link
    def raced(source, destination, *args, **kwargs):
        if Path(destination)==receipt:
            receipt.write_text('other writer')
            raise FileExistsError(receipt)
        return real_link(source,destination,*args,**kwargs)
    monkeypatch.setattr(p,'render',fake_render)
    monkeypatch.setattr(os,'link',raced)
    with pytest.raises(FileExistsError):
        folio_source.main(['--input',str(rendered['src']),'--output',str(output),'--quiet'])
    assert not output.exists() and receipt.read_text()=='other writer'

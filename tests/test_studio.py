"""Studio invariants using synthetic audio; no private recordings or models."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import soundfile as sf
from voicelab import dsp, studio_audio as A, studio_effects as E
from voicelab.studio_pipeline import Renderer

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('folio_studio',ROOT/'tools/folio_studio.py')
import sys
sys.path.insert(0,str(ROOT/'tools'))
cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)


def write(path,x):
    sf.write(path,x,48000,subtype='DOUBLE');return path


def synthetic(seconds=32):
    t=np.arange(48000*seconds)/48000
    rng=np.random.default_rng(42)
    x=(.07+.2*np.sin(2*np.pi*.21*t)**4)*np.sin(2*np.pi*173*t)
    return x+rng.normal(0,.001,len(t))


def test_global_limiter_and_parallel_compressor(tmp_path):
    x=synthetic();x[48000*30:48000*30+5]=.9
    src=write(tmp_path/'input.wav',x)
    A.limit(src,tmp_path/'limited.wav',-12,100)
    expected,_=dsp.limit_peaks(x,peak_db=-12,release_ms=100)
    np.testing.assert_allclose(sf.read(tmp_path/'limited.wav')[0],expected,atol=2e-15)
    A.compress(src,tmp_path/'compressed.wav')
    compressed,_=dsp.compress(x,ratio=3.5,threshold_db=-28,attack_ms=3,release_ms=140,knee_db=8)
    np.testing.assert_allclose(sf.read(tmp_path/'compressed.wav')[0],.2*x+.8*compressed,atol=2e-10)


def test_streamed_edit_and_mapping(tmp_path):
    x=synthetic(34);source=write(tmp_path/'input.wav',x)
    keep=[[0,48000*31],[48000*32,len(x)]];fade=384;edge=384
    edits=dict(keep=keep,crossfade_samples=fade,edge_fade_samples=edge)
    mapping=A.cut(source,tmp_path/'cut.wav',edits)
    expected=x[:keep[0][1]].copy();right=x[keep[1][0]:].copy()
    expected[-fade:]=expected[-fade:]*(1-np.linspace(0,1,fade))+right[:fade]*np.linspace(0,1,fade)
    expected=np.concatenate((expected,right[fade:]))
    expected[:edge]*=np.linspace(0,1,edge);expected[-edge:]*=np.linspace(1,0,edge)
    np.testing.assert_allclose(sf.read(tmp_path/'cut.wav')[0],expected,rtol=0,atol=3e-17)
    assert mapping[1]['output_start_sample']==keep[0][1]-fade
    assert sf.info(tmp_path/'cut.wav').frames==sum(b-a for a,b in keep)-fade


@pytest.mark.parametrize('value',[{'keep':[[2,1]]},{'keep':[[0,10000],[9000,15000]]},{'keep':[[False,10000]]},{'keep':[[0,15000]],'crossfade_samples':float('nan')}])
def test_reject_invalid_edits(value):
    with pytest.raises(ValueError):A.validate_edits(value,20000)


def test_noise_is_deterministic_numeric_profile(tmp_path):
    from scipy import signal
    p=json.loads((ROOT/'presets/studio-v1.json').read_text())['noise'];n=48000*3
    E.noise_bed(tmp_path/'bed.wav',n,p,tmp_path)
    white=np.random.default_rng(p['seed']).standard_normal(n+4096)
    expected=signal.fftconvolve(white,p['taps'],mode='same')[2048:2048+n]
    expected*=10**(p['base_rms_dbfs']/20)/np.sqrt(np.mean(expected**2))
    expected[:384]*=np.linspace(0,1,384);expected[-384:]*=np.linspace(1,0,384)
    np.testing.assert_allclose(sf.read(tmp_path/'bed.wav')[0],expected,atol=2e-18)


def test_cleanup_never_amplifies_and_preserves_clock(tmp_path):
    x=synthetic(3);src=write(tmp_path/'source.wav',x)
    model=write(tmp_path/'model.wav',x*.01)
    E.cleanup(src,src,model,tmp_path/'clean.wav',np.zeros(300,dtype=bool))
    y=sf.read(tmp_path/'clean.wav')[0]
    assert len(y)==len(x)
    # Constant magnitude cap in pauses: the same gain at every frequency/time.
    np.testing.assert_allclose(y,x*10**(-9/20),atol=2e-8)


def test_render_validates_controls_before_work(tmp_path):
    r=Renderer(tmp_path,json.loads((ROOT/'presets/studio-v1.json').read_text()))
    with pytest.raises(ValueError):r.render(None,None,None,blend=float('nan'))


def test_repair_plan_validation():
    assert cli.repair_regions(dict(sample_rate=48000,regions=[[2,4],[6,9]]),10)==[[2,4],[6,9]]
    for pairs in ([[2,11]],[[2,5],[4,6]],[[False,4]]):
        with pytest.raises(ValueError):cli.repair_regions(dict(sample_rate=48000,regions=pairs),10)


def test_existing_output_is_preserved(tmp_path):
    output=tmp_path/'existing.wav';output.write_bytes(b'keep me')
    with pytest.raises(FileExistsError):cli.process(SimpleNamespace(input=tmp_path/'absent.wav',output=output))
    assert output.read_bytes()==b'keep me'


def prepared_cli(tmp_path,monkeypatch):
    src=write(tmp_path/'input.wav',synthetic(5));native=write(tmp_path/'native.wav',synthetic(5)*.5)
    executable=tmp_path/'denoiser';executable.write_bytes(b'test binary')
    native.with_suffix('.json').write_text(json.dumps(dict(family='tf-restormer',
        source_sha256=cli.sha(src),output_sha256=cli.sha(native),
        path='private-input-path',parameters=dict(seconds=3,input_rate=48000,input='private-input-path'))))
    monkeypatch.setattr(cli,'binary',lambda runtime:executable)
    monkeypatch.setattr(cli,'preflight_source',lambda source:None)
    args=SimpleNamespace(input=src,output=tmp_path/'output.wav',native_stem=native,
        preset=ROOT/'presets/studio-v1.json',edit_list=None,repair_regions=None,
        quiet=True,preview_gain_db=None,blend=None,fuzz=None)
    class FakeRenderer:
        def __init__(self,root,*args):self.root=root
        def render(self,source,*args,**kwargs):
            final=self.root/'final.wav';sf.write(final,synthetic(5)*.2,48000,subtype='PCM_24')
            return final,dict(samples=5*48000,reference_audio_read=False)
    monkeypatch.setattr(cli,'Renderer',FakeRenderer)
    return args


def test_cli_publishes_verified_complete_pair(tmp_path,monkeypatch):
    args=prepared_cli(tmp_path,monkeypatch);meta=cli.process(args)
    assert sf.info(args.output).subtype=='PCM_24'
    assert sf.info(args.output).frames==5*48000
    assert meta['source_sha256']==cli.sha(args.input)
    assert meta['output_sha256']==cli.sha(args.output)
    assert 'path' not in meta['native_restoration']
    assert 'input' not in meta['native_restoration']['parameters']
    assert json.loads(args.output.with_suffix('.json').read_text())==meta
    assert not list(tmp_path.glob('.folio-studio-*'))


def test_cli_rolls_back_own_audio_when_receipt_publish_fails(tmp_path,monkeypatch):
    args=prepared_cli(tmp_path,monkeypatch);original_link=cli.os.link
    receipt=args.output.with_suffix('.json')
    def link(source,dest):
        if dest==receipt:
            receipt.write_text('another writer');raise FileExistsError('raced receipt')
        original_link(source,dest)
    monkeypatch.setattr(cli.os,'link',link)
    with pytest.raises(FileExistsError):cli.process(args)
    assert not args.output.exists()
    assert receipt.read_text()=='another writer'
    assert not list(tmp_path.glob('.folio-studio-*'))


def test_cli_rejects_cached_native_for_another_source(tmp_path,monkeypatch):
    args=prepared_cli(tmp_path,monkeypatch)
    write(args.input,synthetic(5)*.9)
    with pytest.raises(ValueError,match='Cached native stem'):cli.process(args)
    assert not args.output.exists()
    assert not list(tmp_path.glob('.folio-studio-*'))


def test_cli_interrupt_cleans_temporary_render(tmp_path,monkeypatch):
    args=prepared_cli(tmp_path,monkeypatch)
    def interrupted(*args,**kwargs):raise KeyboardInterrupt
    monkeypatch.setattr(cli.Renderer,'render',interrupted)
    with pytest.raises(KeyboardInterrupt):cli.process(args)
    assert not args.output.exists()
    assert not args.output.with_suffix('.json').exists()
    assert not list(tmp_path.glob('.folio-studio-*'))


def test_interrupted_model_is_terminated_before_cleanup(monkeypatch):
    calls=[]
    class Child:
        def __init__(self,*a):pass
        def wait(self,timeout=None):
            if timeout is None:raise KeyboardInterrupt
            calls.append('wait')
        def poll(self):return None
        def terminate(self):calls.append('terminate')
    monkeypatch.setattr(E.subprocess,'Popen',Child)
    with pytest.raises(KeyboardInterrupt):E.run_child(['local-model'])
    assert calls==['terminate','wait']


def test_robust_speech_survives_one_loud_transient(tmp_path):
    t=np.arange(12*48000)/48000
    active=(t%2)<1.5
    x=.002*np.sin(2*np.pi*173*t)*active+np.random.default_rng(4).normal(0,.000005,len(t))
    x[48000*5:48000*5+4800]=.99
    src=write(tmp_path/'transient.wav',x)
    assert A.mask(src).mean()<.1
    mask=A.robust_speech_mask(src)
    assert .5<mask.mean()<.9
    warm=write(tmp_path/'warm.wav',x*.1)
    gain=A.preview_gain(src,warm)
    assert 30<gain<=36


def test_pcm_guard_refuses_clipped_intermediate(tmp_path):
    source=write(tmp_path/'input.wav',np.array([0.,1.2,0.]))
    with pytest.raises(ValueError,match='would clip'):
        A.gain(source,tmp_path/'unsafe.wav',0,'PCM_24')


def test_partly_collapsed_mask_uses_robust_activity(tmp_path,monkeypatch):
    monkeypatch.setattr(A,'mask',lambda source:np.arange(100)<25)
    monkeypatch.setattr(A,'robust_speech_mask',lambda source:np.arange(100)<70)
    mask,details=A.activity_mask('unused')
    assert mask.sum()==70
    assert details['original_speech_fraction']==.25
    assert details['speech_mask_method']=='robust-percentile-fallback'
    monkeypatch.setattr(A,'mask',lambda source:np.arange(100)<63)
    mask,details=A.activity_mask('unused')
    assert mask.sum()==63
    assert details['speech_mask_method']=='original-energy'

"""Parallel windows, stem release, tiers and segmented denoising.

The speedups must not change the approved render: with any worker count each
window is computed by the same function on the same samples.
"""
import hashlib
import json
from pathlib import Path
import stat
import sys

import numpy as np
import pytest
import soundfile as sf

from voicelab import chunks, dsp, parallel, source_pipeline, streaming, tiers
from voicelab import studio_audio as A, studio_effects as E
from voicelab.f_master_file import master_file
from voicelab.studio_pipeline import Renderer
from synthetic_speech import narration

ROOT = Path(__file__).parents[1]
RATE = 48000


def digest(path):
    """Hash of the decoded samples. Float WAV headers carry a timestamp, so
    file bytes differ between two otherwise identical intermediate stems."""
    x, _ = sf.read(path, dtype='float64')
    return hashlib.sha256(x.tobytes()).hexdigest()


def take(seconds=40, seed=3):
    return narration(seconds=seconds, seed=seed)[0]


def write(path, x, subtype='PCM_24'):
    sf.write(path, x, RATE, subtype=subtype)
    return path


@pytest.fixture
def pool():
    def configure(workers):
        return parallel.configure(workers)
    yield configure
    parallel.configure(1)


def test_ordered_map_matches_inline(pool):
    tasks = [(np.arange(5)+i, i) for i in range(23)]
    pool(1)
    inline = list(parallel.imap(np.multiply, tasks))
    pool(3)
    pooled = list(parallel.imap(np.multiply, tasks))
    assert all(np.array_equal(a, b) for a, b in zip(inline, pooled))
    assert len(pooled) == 23
    # A closure stays inline when a stage opts out of the pool.
    assert list(parallel.imap(lambda x, y: x+y, [(1, 2), (3, 4)], enabled=False)) == [3, 7]


def test_true_peak_meter_is_identical_with_workers(pool):
    x = take(12)
    def measure():
        meter = streaming.TruePeakMeter()
        for start in range(0, len(x), 50000):
            meter.update(x[start:start+50000])
        return meter.finish()
    pool(1)
    serial = measure()
    pool(3)
    assert measure() == serial


def test_source_pipeline_is_byte_identical_with_workers(tmp_path, pool):
    source = write(tmp_path/'take.wav', take(70))
    preset = json.loads((ROOT/'presets/source-narration-v1.json').read_text())
    digests = []
    for workers in (1, 3):
        pool(workers)
        work = tmp_path/f'work-{workers}'
        work.mkdir()
        final, details = source_pipeline.render(source, work, preset)
        digests.append((digest(final), json.dumps(details, sort_keys=True)))
    assert digests[0] == digests[1]


def test_master_file_is_byte_identical_with_workers(tmp_path, pool):
    source = write(tmp_path/'native.wav', take(65)*.6)
    preset = json.loads((ROOT/'presets/studio-v1.json').read_text())['early_master']
    digests = []
    for workers in (1, 3):
        pool(workers)
        target = tmp_path/f'mastered-{workers}.wav'
        master_file(source, target, preset)
        digests.append(digest(target))
    assert digests[0] == digests[1]


def test_cleanup_and_tone_are_byte_identical_with_workers(tmp_path, pool):
    x = take(65)
    source = write(tmp_path/'selected.wav', x*.5, 'FLOAT')
    guide = write(tmp_path/'guide.wav', x*.45, 'FLOAT')
    model = write(tmp_path/'model.wav', x*.3, 'FLOAT')
    speech = A.mask(source)
    profile = json.loads((ROOT/'presets/studio-v1.json').read_text())['ordinary_tone']
    digests = []
    for workers in (1, 3):
        pool(workers)
        cleaned = E.cleanup(source, guide, model, tmp_path/f'cleaned-{workers}.wav', speech)
        toned = Renderer(tmp_path, {}).tone(source, f'toned-{workers}', profile)
        digests.append((digest(cleaned), digest(toned)))
    assert digests[0] == digests[1]


def test_rms_only_statistics_match_full_statistics(tmp_path):
    path = write(tmp_path/'take.wav', take(8), 'DOUBLE')
    speech = A.mask(path)
    full = A.stats(path, speech)
    quick = A.stats(path, speech, true_peak=False)
    assert quick['true_peak'] is None
    assert (quick['rms'], quick['peak'], quick['samples']) == (full['rms'], full['peak'], full['samples'])


def test_release_removes_only_finished_stems(tmp_path):
    renderer = Renderer(tmp_path, {})
    kept = renderer.path('kept')
    finished = renderer.path('finished')
    for path in (kept, finished, tmp_path/'native.wav'):
        write(path, np.zeros(4800))
    renderer.release(kept)
    assert kept.exists() and not finished.exists()
    assert (tmp_path/'native.wav').exists()  # the model's stem is not the renderer's to remove
    assert renderer.peak_bytes >= 3*kept.stat().st_size


def fake_denoiser(tmp_path):
    """A stand-in executable that returns its input, like a 0 dB denoiser."""
    script = tmp_path/'denoiser'
    script.write_text(f'#!{sys.executable}\n'
                      'import shutil,sys,pathlib\n'
                      'out=pathlib.Path(sys.argv[sys.argv.index("-o")+1]);out.mkdir(parents=True,exist_ok=True)\n'
                      'shutil.copy(sys.argv[-1],out/pathlib.Path(sys.argv[-1]).name)\n')
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return script


def test_segmented_denoise_keeps_clock_and_joins(tmp_path, monkeypatch):
    monkeypatch.setattr(E, 'DENOISE_MINIMUM', 4*RATE)
    monkeypatch.setattr(E, 'DENOISE_PREROLL', RATE)
    monkeypatch.setattr(E, 'DENOISE_POSTROLL', RATE//2)
    x = take(21).astype('float32')
    source = write(tmp_path/'pre.wav', x, 'FLOAT')
    binary = fake_denoiser(tmp_path)
    single = E.denoise(source, tmp_path/'single.wav', binary, tmp_path/'one')
    several = E.denoise(source, tmp_path/'several.wav', binary, tmp_path/'many', segments=4)
    assert 'segments' not in single and several['segments'] == 4
    assert several['segment_edges'][0] == 0 and several['segment_edges'][-1] == len(x)
    a, _ = sf.read(tmp_path/'single.wav', dtype='float32')
    b, _ = sf.read(tmp_path/'several.wav', dtype='float32')
    assert np.array_equal(a, x)
    assert len(b) == len(x) and np.max(abs(b-x)) < 1e-6
    assert not (tmp_path/'one').exists() and not (tmp_path/'many').exists()


def test_short_takes_use_the_single_denoise_pass():
    assert E.denoise_plan(100*RATE, 8) == [0, 100*RATE]
    assert E.denoise_plan(600*RATE, 4) == [0, 150*RATE, 300*RATE, 450*RATE, 600*RATE]
    assert len(E.denoise_plan(600*RATE, 8)) == 6  # never shorter than two minutes


def test_tiers_and_saved_stems():
    reference = tiers.resolve('reference', 'cuda')
    assert reference['model_rate'] == 48000 and reference['precision'] == 'float32'
    assert not reference['denoise_segments'] and not reference['skip_silence']
    assert tiers.resolve('reference', 'cpu')['forward'] == 'lean'
    # A v0.3 stem has none of the new fields and is the reference.
    assert tiers.stem_matches({'seconds': 3, 'input_rate': 48000}, reference)
    assert not tiers.stem_matches({'model_input_rate': 16000}, reference)
    assert not tiers.stem_matches({'forward': 'sliced'}, reference)
    fast = tiers.resolve('fast', 'cpu')
    assert tiers.stem_matches({'model_input_rate': 16000, 'skipped_windows': 3, 'forward': 'sliced'}, fast)
    assert not tiers.stem_matches({'model_input_rate': 48000}, fast)
    with pytest.raises(ValueError):
        tiers.resolve('ludicrous', 'cpu')


def test_restore_file_reports_window_positions(tmp_path):
    x = take(8)
    source = write(tmp_path/'in.wav', x, 'DOUBLE')
    seen = []
    def infer(block, start, end):
        seen.append((start, end))
        return np.zeros(len(block)) if start == 3*RATE else block
    chunks.restore_file(source, tmp_path/'out.wav', infer, with_positions=True)
    assert seen == [(0, 3*RATE), (3*RATE, 6*RATE), (6*RATE, 8*RATE)]
    y, _ = sf.read(tmp_path/'out.wav')
    assert np.array_equal(y[:2*RATE], x[:2*RATE])
    assert np.max(abs(y[3*RATE+4000:6*RATE-4000])) == 0


def test_lean_attention_and_slices_match_stock_blocks():
    torch = pytest.importorskip('torch')
    from voicelab import tf_fast
    torch.manual_seed(5)
    query, key, value = (torch.randn(3, 4, 17, 8) for _ in range(3))
    stock = torch.nn.functional.scaled_dot_product_attention(query, key, value)
    assert torch.allclose(tf_fast._attention(query, key, value), stock, atol=1e-6)
    block = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.SiLU())
    x = torch.randn(11, 5, 8)
    with torch.inference_mode():
        assert torch.allclose(tf_fast.Sliced(block, 4)(x), block(x), atol=1e-6)
        assert torch.equal(tf_fast.Sliced(block, 64)(x), block(x))


def test_model_process_does_not_inherit_serial_library_limits(monkeypatch):
    monkeypatch.delenv('OMP_NUM_THREADS', raising=False)
    monkeypatch.setenv('MKL_NUM_THREADS', '3')
    monkeypatch.setattr(parallel, '_pinned', [])
    parallel.serial_numeric_libraries()
    import os
    assert os.environ['OMP_NUM_THREADS'] == '1'
    environment = parallel.model_environment()
    assert 'OMP_NUM_THREADS' not in environment      # pinned here, so not passed on
    assert environment['MKL_NUM_THREADS'] == '3'     # set by the user, so kept
    for name in parallel._pinned:
        monkeypatch.delenv(name, raising=False)

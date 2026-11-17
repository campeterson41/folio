"""Measured final mastering for narration; preserves the performance's clock.

master_me supplies adaptive leveling, compression and limiting. FFmpeg's EBU
R128 meter supplies independent final loudness verification and gain trim. The
conservative -26 LUFS mono target is a local listening choice, not a delivery
certification. Existing Studio vocal compression remains upstream.
"""
import json
import math
from pathlib import Path
import shutil
import subprocess
import tempfile

import soundfile as sf
from . import studio_audio as A

REVISION = 'master-me-narration-v1'
TARGET_LUFS = -26.
TRUE_PEAK_DB = -3.


def require_meter():
    executable = shutil.which('ffmpeg')
    if executable is None:
        raise ValueError('Measured Studio mastering requires FFmpeg on PATH')
    return executable


def measure(source):
    # A disk-backed log avoids pipe deadlocks and unbounded stderr buffering.
    with tempfile.TemporaryFile(mode='w+') as log:
        child = subprocess.Popen([require_meter(), '-hide_banner', '-nostats',
            '-i', str(source), '-af',
            'loudnorm=I=-26:TP=-3:LRA=50:print_format=json',
            '-f', 'null', '-'], stdout=subprocess.DEVNULL, stderr=log)
        try:
            if child.wait():
                raise ValueError('FFmpeg loudness measurement failed')
        except BaseException:
            if child.poll() is None:
                child.terminate()
                try: child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill(); child.wait()
            raise
        log.seek(0)
        report = log.read()
    try:
        values = json.JSONDecoder().raw_decode(report[report.rfind('{'):])[0]
        result = {key: float(values[name]) for key, name in
            [('integrated_lufs', 'input_i'), ('true_peak_db', 'input_tp'),
             ('loudness_range_lu', 'input_lra')]}
    except (ValueError, KeyError) as exc:
        raise ValueError('Invalid loudness measurement') from exc
    if not all(math.isfinite(v) for v in result.values()):
        raise ValueError('No measurable programme loudness; mastering refused')
    return result


def master(source, target):
    source, target = Path(source), Path(target)
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    info = sf.info(source)
    if info.samplerate != A.RATE or info.channels != 1:
        raise ValueError('Mastering requires 48 kHz mono')
    from . import master_me
    master_me.engine()
    initial = measure(source)
    current = source
    total_gain = 0.
    reductions = []
    with tempfile.TemporaryDirectory(prefix='.master-', dir=target.parent) as tmp:
        # Bring the whole take near target before the adaptive engine starts,
        # preventing the opening phrase from bearing the full input gain error.
        prematch=min(TARGET_LUFS-initial['integrated_lufs'],12.)
        matched=Path(tmp)/'matched.wav';A.gain(source,matched,prematch)
        total_gain=prematch
        current=Path(tmp)/'master-me.wav'
        engine_info=master_me.process(matched,current)
        if sf.info(current).frames!=info.frames:
            raise ValueError('master_me changed the sample clock')
        for i in range(2):
            measured = measure(current)
            # Do not turn virtually silent material into a loud noise floor.
            db = min(TARGET_LUFS-measured['integrated_lufs'], 12.-total_gain)
            gained = Path(tmp)/f'gain-{i}.wav'
            A.gain(current, gained, db)
            total_gain += db
            limited = Path(tmp)/f'limit-{i}.wav'
            details = A.limit(gained, limited, TRUE_PEAK_DB-.2, 100)
            reductions.append(details['maximum_reduction_db'])
            current = limited
            if reductions[-1] < .01:
                break
        # Independent oversampled verification catches intersample overshoot.
        peak = 20*math.log10(max(A.stats(current)['true_peak'], 1e-12))
        safety = min(0., TRUE_PEAK_DB-.02-peak)
        staged = Path(tmp)/'verified.wav'
        A.gain(current, staged, safety, 'PCM_24')
        final = measure(staged)
        stats = A.stats(staged)
        if (sf.info(staged).frames != info.frames or
                stats['true_peak'] > 10**(TRUE_PEAK_DB/20) or
                final['true_peak_db'] > TRUE_PEAK_DB+.01):
            raise ValueError('Mastering failed clock/true-peak verification')
        # Never publish incomplete audio or replace an existing file.
        import os
        os.link(staged, target)
    return dict(revision=REVISION, target_lufs=TARGET_LUFS,
        true_peak_ceiling_db=TRUE_PEAK_DB, input=initial, output=final,
        uniform_gain_db=total_gain+safety,
        limiter_reduction_each_pass_db=reductions,
        target_reached=abs(final['integrated_lufs']-TARGET_LUFS)<=.3,
        samples=info.frames, engine=engine_info, meter='FFmpeg EBU R128 (mono, dual_mono=false)')

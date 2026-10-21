#!/usr/bin/env python3
"""Folio source pipeline: CPU-only narration cleanup from the recording itself.

Needs only NumPy, SciPy and SoundFile. No model, reference recording, network
call, transcript or speech editing is used. Input must be 48 kHz mono WAV.
"""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import tempfile

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from voicelab import __version__  # noqa: E402
from voicelab import source_pipeline as pipeline  # noqa: E402
from voicelab.streaming import sha  # noqa: E402

RECIPE = 'folio-source-v1'
DEFAULT_PRESET = ROOT/'presets/source-narration-v1.json'
MINIMUM_SECONDS = 5
IMPLEMENTATION = [Path(__file__).resolve(), ROOT/'src/voicelab/source_pipeline.py',
                  ROOT/'src/voicelab/dsp.py', ROOT/'src/voicelab/streaming.py',
                  ROOT/'src/voicelab/adaptive.py']


def inspect_input(path, block=1048576):
    """Bounded scan: format, finiteness and near-full-scale samples."""
    info = sf.info(path)
    if info.format not in ('WAV', 'WAVEX', 'RF64'):
        raise ValueError('Input must be a WAV file')
    if info.samplerate != pipeline.RATE or info.channels != 1:
        raise ValueError('Input must be 48 kHz mono; convert it first '
                         f'(found {info.samplerate} Hz, {info.channels} channels)')
    if info.frames < MINIMUM_SECONDS*pipeline.RATE:
        raise ValueError(f'Input is shorter than {MINIMUM_SECONDS} seconds')
    peak, near_full = 0., 0
    with sf.SoundFile(path) as f:
        for x in f.blocks(block, dtype='float64'):
            if not np.isfinite(x).all():
                raise ValueError('Input contains non-finite samples')
            peak = max(peak, float(np.max(abs(x), initial=0)))
            near_full += int(np.count_nonzero(abs(x) >= .999))
    if peak == 0:
        raise ValueError('Input is silent')
    return info, {'frames': info.frames, 'subtype': info.subtype, 'peak': peak,
                  'samples_at_or_above_minus_0_01_dbfs': near_full}


def _interrupt(signum, frame):
    raise KeyboardInterrupt(f'Stopped by signal {signum}')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input', type=Path, required=True, help='48 kHz mono WAV')
    ap.add_argument('--output', type=Path, required=True,
                    help='New .wav path; a .json receipt is written beside it')
    ap.add_argument('--preset', type=Path, default=DEFAULT_PRESET)
    ap.add_argument('--tail-db', type=float,
                    help='Late-tail attenuation cap, 0–6 dB (preset default 6)')
    ap.add_argument('--bass-db', type=float,
                    help='160 Hz low-shelf gain, 0–4 dB (preset default 3)')
    ap.add_argument('--quiet', action='store_true', help='No progress output')
    args = ap.parse_args(argv)
    preset = json.loads(args.preset.read_text())
    tail = preset.get('tail_db', 6.) if args.tail_db is None else args.tail_db
    bass = preset.get('bass_db', 3.) if args.bass_db is None else args.bass_db
    if not (isinstance(tail, (int, float)) and math.isfinite(tail)
            and 0 <= tail <= pipeline.MAXIMUM_TAIL_DB):
        ap.error('--tail-db must be between 0 and 6 dB')
    if not (isinstance(bass, (int, float)) and math.isfinite(bass)
            and 0 <= bass <= pipeline.MAXIMUM_BASS_DB):
        ap.error('--bass-db must be between 0 and 4 dB')
    output, receipt = args.output, args.output.with_suffix('.json')
    if output.suffix.lower() != '.wav':
        ap.error('Use a .wav output path')
    if not args.input.is_file():
        ap.error(f'Input not found: {args.input}')
    source = args.input.resolve()
    if source in (output.resolve(), receipt.resolve()):
        ap.error('Output and receipt must differ from the input')
    output.parent.mkdir(parents=True, exist_ok=True)
    previous = {s: signal.signal(s, _interrupt) for s in (signal.SIGTERM, signal.SIGHUP)}
    try:
        lock_path = output.parent/('.'+output.name+'.lock')
        with lock_path.open('a') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise FileExistsError('This output is already being processed')
            try:
                process(source, output, receipt, args.preset, preset, float(tail),
                        float(bass), args.quiet)
            finally:
                # Removed while still held; a late contender that locks the old
                # inode still cannot publish over the exclusive hard links.
                lock_path.unlink(missing_ok=True)
    finally:
        for s, handler in previous.items():
            signal.signal(s, handler)
    if not args.quiet:
        print(output)


def process(source, output, receipt, preset_path, preset, tail, bass, quiet=False):
    for p in (output, receipt):
        if p.exists() or p.is_symlink():
            raise FileExistsError(f'Refusing to overwrite {p}')
    info, input_check = inspect_input(source)
    source_hash = sha(source)
    warnings = []
    if input_check['samples_at_or_above_minus_0_01_dbfs']:
        warnings.append('Input has samples at full scale; clipping in the recording '
                        'cannot be repaired by this pipeline.')

    def progress(phase, seconds):
        if not quiet:
            print(f'\r{phase:<26} {seconds/60:5.1f} of {info.frames/pipeline.RATE/60:.1f} min',
                  end='', file=sys.stderr, flush=True)

    with tempfile.TemporaryDirectory(prefix='.folio-source-', dir=output.parent) as tmp:
        final, details = pipeline.render(source, tmp, preset, tail, bass, progress)
        if not quiet:
            print(file=sys.stderr)
        # Independent read-back of the quantized file before publication.
        check = sf.info(final)
        peak = 0.
        with sf.SoundFile(final) as f:
            for x in f.blocks(1048576, dtype='float64'):
                if not np.isfinite(x).all():
                    raise ValueError('Non-finite output')
                peak = max(peak, float(np.max(abs(x), initial=0)))
        if (check.frames != info.frames or check.samplerate != pipeline.RATE
                or check.channels != 1 or check.subtype != 'PCM_24' or peak >= .999):
            raise ValueError('Rendered output failed final length/format/peak checks')
        if sha(source) != source_hash:
            raise RuntimeError('Input changed during processing; nothing was published')
        for side, value in details['limiter_max_reduction_db'].items():
            if value > 6:
                warnings.append(f'Peak limiter reduced loud passages by up to {value:.1f} dB '
                                f'in the {side} branch; audition loud moments.')
        files = {str(p.relative_to(ROOT)): sha(p) for p in IMPLEMENTATION}
        metadata = {
            'recipe': RECIPE, 'folio_version': __version__,
            'source_sha256': source_hash, 'output_sha256': sha(final),
            'samples': info.frames, 'sample_rate': pipeline.RATE,
            'input_check': input_check, 'output_peak': peak,
            'preset_sha256': sha(preset_path), 'preset_label': preset.get('label'),
            'implementation_sha256': files,
            'implementation_digest': hashlib.sha256(
                json.dumps(files, sort_keys=True).encode()).hexdigest(),
            **details,
            'neural_models_used': False, 'reference_audio_read': False,
            'network_used': False, 'speech_edited': False, 'warnings': warnings,
            'status': 'Experimental. Tail reduction is a fixed heuristic and can '
                      'affect quiet speech; audition before use.',
        }
        meta = Path(tmp)/'receipt.json'
        meta.write_text(json.dumps(metadata, indent=2)+'\n')
        # Exclusive hard links publish complete files and never replace a file
        # that appeared while processing. The temporary directory shares the FS.
        try:
            os.link(final, output)
            os.link(meta, receipt)
        except BaseException:
            # link() can succeed just before an interrupt reaches Python.
            # Roll back only inodes we created; preserve another writer's file.
            for destination, original in ((output, final), (receipt, meta)):
                try:
                    if os.path.samefile(destination, original):
                        destination.unlink()
                except FileNotFoundError:
                    pass
            raise
    for w in warnings:
        print('Warning: '+w, file=sys.stderr)
    return metadata


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nInterrupted; temporary files removed and nothing published.', file=sys.stderr)
        sys.exit(130)
    except (ValueError, OSError, RuntimeError, sf.LibsndfileError) as e:
        print(f'folio-source: {e}', file=sys.stderr)
        sys.exit(1)

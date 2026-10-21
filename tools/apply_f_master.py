#!/usr/bin/env python3
"""Apply the frozen F tone/dynamics and preferred presence EQ to a native stem.

This does not run restoration or consult reference audio. The input must already
be a TF-Restormer native output. Original S-detail experiments are separate.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

from voicelab import dsp


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def apply(x, preset):
    y, metrics = dsp.master(
        x, profile=preset['tone_profile'], **preset['mastering_settings']
    )
    # F was stored as PCM24 before the final EQ trial. Preserve that intermediate
    # quantization so this reproduces the accepted short-take signal exactly.
    intermediate = io.BytesIO()
    sf.write(intermediate, y, dsp.RATE, format='WAV', subtype='PCM_24')
    intermediate.seek(0)
    y, _ = sf.read(intermediate, dtype='float64')
    eq = preset['preferred_final_eq']
    if eq['type'] != 'peaking_biquad':
        raise ValueError('Unsupported final EQ')
    A = 10 ** (eq['gain_db'] / 40)
    w = 2 * np.pi * eq['hz'] / dsp.RATE
    alpha = np.sin(w) / (2 * eq['q'])
    b = np.array([1 + alpha*A, -2*np.cos(w), 1 - alpha*A])
    a = np.array([1 + alpha/A, -2*np.cos(w), 1 - alpha/A])
    y = signal.lfilter(b / a[0], a / a[0], y)
    metrics['post_eq_sample_peak_db'] = float(dsp.db(np.max(abs(y), initial=0)))
    metrics['post_eq_rms_db'] = float(dsp.db(dsp.rms(y)))
    return y, metrics


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--preset', type=Path, default=Path(
        'presets/f-voice-tone-20261021.json'))
    args = ap.parse_args()
    receipt = args.output.with_suffix('.json')
    if args.output.exists() or receipt.exists():
        raise FileExistsError('Refusing to replace an existing render or receipt')
    info = sf.info(args.input)
    if info.samplerate != dsp.RATE or info.channels != 1:
        raise ValueError('Expected a native 48 kHz mono restoration stem')
    preset = json.loads(args.preset.read_text())
    y, metrics = apply(dsp.load(args.input), preset)
    if len(y) != info.frames:
        raise ValueError('Mastering changed sample count')
    dsp.save(args.output, y)
    if sf.info(args.output).frames != info.frames:
        raise ValueError('Saved audio changed sample count')
    receipt.write_text(json.dumps({
        'input': str(args.input.resolve()), 'input_sha256': sha(args.input),
        'output_sha256': sha(args.output), 'preset': str(args.preset.resolve()),
        'preset_sha256': sha(args.preset), 'samples': len(y),
        'sample_rate': dsp.RATE, 'metrics': metrics,
        'note': 'Fixed F settings plus preferred presence EQ; no refitting, '
                'reference audio, or S-detail preservation. Whole-file CPU mastering.',
    }, indent=2) + '\n')
    print(receipt)


if __name__ == '__main__':
    main()

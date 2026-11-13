#!/usr/bin/env python3
"""Time Folio Studio's quality tiers on one take and measure how far each
render is from the reference render.

No listening test is implied. The distances say how much a faster tier changed
the approved sound, not whether the change is better or worse.

    tools/benchmark_studio.py --input take.wav --workdir bench --device cpu

Pass ``--baseline /path/to/older/checkout`` to also time that checkout's
``bin/folio`` and compare the reference tier against it.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import numpy as np
import soundfile as sf
from scipy import signal
from voicelab import dsp, tiers

RATE = 48000
BANDS = [(63, 125), (125, 250), (250, 500), (500, 1000), (1000, 2000), (2000, 4000),
         (4000, 8000), (8000, 16000)]


def distance(reference, candidate):
    """Objective distance of ``candidate`` from ``reference`` (both 48 kHz mono)."""
    r, rate = sf.read(reference, dtype='float64')
    y, other = sf.read(candidate, dtype='float64')
    if rate != RATE or other != RATE or r.ndim != 1 or y.ndim != 1 or len(r) != len(y):
        raise ValueError('Renders must be 48 kHz mono with the same sample count')
    if np.array_equal(r, y):
        return {'identical_samples': True, 'residual_db': None, 'log_spectral_distance_db': 0.,
                'largest_band_shift_db': 0., 'speech_level_shift_db': 0., 'pause_level_shift_db': 0.}
    speech = dsp.speech_mask_energy(dsp.envelope(r))
    samples = np.repeat(speech, 480)[:len(r)]
    samples = np.pad(samples, (0, len(r)-len(samples)))
    power = lambda x: float(np.mean(x*x))+1e-30
    _, _, zr = signal.stft(r, RATE, nperseg=2048, noverlap=1536)
    _, _, zy = signal.stft(y, RATE, nperseg=2048, noverlap=1536)
    frames = np.interp(np.arange(zr.shape[1])*512/480, np.arange(len(speech)), speech) > .5
    f = np.fft.rfftfreq(2048, 1/RATE)
    audible = (f >= 50) & (f <= 20000)
    floor = np.max(abs(zr))**2*1e-10  # -100 dB re the loudest bin
    lr = 10*np.log10(np.maximum(abs(zr[audible][:, frames])**2, floor))
    ly = 10*np.log10(np.maximum(abs(zy[audible][:, frames])**2, floor))
    lsd = float(np.mean(np.sqrt(np.mean((lr-ly)**2, axis=0))))
    shifts = {}
    for low, high in BANDS:
        band = (f >= low) & (f < high)
        shifts[f'{low}-{high}'] = float(10*np.log10(
            (np.mean(abs(zy[band][:, frames])**2)+1e-30)/(np.mean(abs(zr[band][:, frames])**2)+1e-30)))
    return {'identical_samples': False,
            'residual_db': float(10*np.log10(power(y-r)/power(r))),
            'log_spectral_distance_db': lsd,
            'largest_band_shift_db': float(max(abs(v) for v in shifts.values())),
            'band_shift_db': shifts,
            'speech_level_shift_db': float(10*np.log10(power(y[samples])/power(r[samples]))),
            'pause_level_shift_db': float(10*np.log10(power(y[~samples])/power(r[~samples])))}


def render(checkout, source, output, device, tier=None, extra=()):
    for path in (output, output.with_suffix('.json')):
        path.unlink(missing_ok=True)
    command = [str(Path(checkout)/'bin/folio'), '--input', str(source), '--output', str(output),
               '--device', device, '--quiet', *extra]
    if tier:
        command += ['--quality', tier]
    start = time.monotonic()
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
    wall = time.monotonic()-start
    receipt = json.loads(output.with_suffix('.json').read_text())
    return {'wall_seconds': wall, 'model_seconds': receipt['native_restoration']['runtime_seconds'],
            'temporary_peak_bytes': receipt.get('temporary_peak_bytes'),
            'output_sha256': receipt['output_sha256']}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--input', type=Path, required=True)
    ap.add_argument('--workdir', type=Path, required=True)
    ap.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    ap.add_argument('--tiers', default=','.join(tiers.TIERS))
    ap.add_argument('--baseline', type=Path, help='Another Folio checkout to time and compare')
    ap.add_argument('--workers', type=int)
    args = ap.parse_args()
    names = [name for name in args.tiers.split(',') if name]
    for name in names:
        tiers.resolve(name, args.device)
    args.workdir.mkdir(parents=True, exist_ok=True)
    seconds = sf.info(args.input).duration
    extra = ['--workers', str(args.workers)] if args.workers else []
    results = {'input_seconds': seconds, 'device': args.device, 'cpus': os.cpu_count(), 'runs': {}}
    report = args.workdir/'benchmark.json'

    def record(name, run):
        run['seconds_per_audio_minute'] = run['wall_seconds']/seconds*60
        results['runs'][name] = run
        report.write_text(json.dumps(results, indent=2)+'\n')
        print(f"{name:10s} {run['wall_seconds']:8.1f} s  ({run['seconds_per_audio_minute']:.1f} s per audio minute)",
              flush=True)

    if args.baseline:
        record('baseline', render(args.baseline, args.input, args.workdir/'baseline.wav', args.device))
    for name in names:
        try:
            record(name, render(ROOT, args.input, args.workdir/f'{name}.wav', args.device, name, extra))
        except subprocess.CalledProcessError as error:
            # Keep timing the other tiers; a failed tier is reported, not hidden.
            results.setdefault('failed', {})[name] = error.returncode
            print(f'{name:10s} failed with exit status {error.returncode}', flush=True)
    if not results['runs']:
        raise SystemExit('No tier rendered')
    anchor = args.workdir/(f'{next(iter(results["runs"]))}.wav')
    for name, run in results['runs'].items():
        run['distance_from'] = anchor.stem
        run['distance'] = distance(anchor, args.workdir/f'{name}.wav')
    report.write_text(json.dumps(results, indent=2)+'\n')
    print(f'\n| run | wall s | s per audio min | residual dB | log-spectral dB | largest band shift dB |')
    print('| --- | ---: | ---: | ---: | ---: | ---: |')
    for name, run in results['runs'].items():
        d = run['distance']
        residual = 'identical' if d['identical_samples'] else f"{d['residual_db']:.1f}"
        print(f"| {name} | {run['wall_seconds']:.1f} | {run['seconds_per_audio_minute']:.1f} | {residual} | "
              f"{d['log_spectral_distance_db']:.2f} | {d['largest_band_shift_db']:.2f} |")
    print(report)


if __name__ == '__main__':
    main()

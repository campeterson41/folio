"""Bounded-memory, multi-pass chapter processing with context at chunk boundaries."""

from __future__ import annotations
import hashlib, math
import numpy as np
import soundfile as sf
from scipy import signal
from . import dsp


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def windows(path, chunk_seconds=30, context_seconds=4):
    with sf.SoundFile(path) as f:
        count = len(f)
        rate = f.samplerate
        chunk = round(chunk_seconds * rate)
        context = round(context_seconds * rate)
        for start in range(0, count, chunk):
            end = min(count, start + chunk)
            a = max(0, start - context)
            b = min(count, end + context)
            f.seek(a)
            x = f.read(b - a, dtype="float64", always_2d=True).mean(axis=1)
            yield start, end, x, slice(start - a, end - a)


def speech_rms_file(path, chunk_seconds=30):
    """Global speech level from 10ms frames, bounded to a few MB per chapter."""
    energy = []
    hop = 480
    for _, _, x, crop in windows(path, chunk_seconds, 4):
        e = dsp.envelope(x)
        energy.append(e[crop.start // hop : math.ceil(crop.stop / hop)])
    e = np.concatenate(energy) if energy else np.empty(0)
    mask = dsp.speech_mask_energy(e)
    return float(dsp.db(np.sqrt(np.mean(e[mask] ** 2)))) if mask.any() else None


class TruePeakMeter:
    """4x peak estimate across block joins with real context, no boundary zero padding."""

    def __init__(self):
        self.tail = np.empty(0)
        self.started = False
        self.peak = 0.0

    def update(self, x):
        joined = np.concatenate((self.tail, x))
        if len(joined) <= 64:
            self.tail = joined.copy()
            return
        high = signal.resample_poly(joined, 4, 1)
        a = 128 if self.started else 0
        b = (len(joined) - 32) * 4
        self.peak = max(self.peak, float(np.max(abs(high[a:b]), initial=0)))
        self.tail = joined[-64:].copy()
        self.started = True

    def finish(self):
        if len(self.tail):
            high = signal.resample_poly(self.tail, 4, 1)
            a = 128 if self.started else 0
            self.peak = max(self.peak, float(np.max(abs(high[a:]), initial=0)))
        return self.peak


def stage(
    source,
    target,
    transform,
    progress,
    phase,
    chunk_seconds,
    context_seconds=4,
    crossfade_seconds=0.0,
    with_position=False,
    subtype="FLOAT",
):
    energy = 0.0
    peak = 0.0
    count = 0
    meter = TruePeakMeter()
    pending = None
    total = sf.info(source).frames
    fade = round(crossfade_seconds * dsp.RATE)
    if fade > round(context_seconds * dsp.RATE):
        raise ValueError("Crossfade requires enough model context")
    with sf.SoundFile(
        target, "w", samplerate=dsp.RATE, channels=1, subtype=subtype, format="RF64"
    ) as out:

        def emit(y):
            nonlocal energy, peak, count
            if not np.isfinite(y).all():
                raise ValueError("Non-finite output in " + phase)
            out.write(y)
            energy += np.dot(y, y)
            peak = max(peak, float(np.max(abs(y), initial=0)))
            count += len(y)
            meter.update(y)

        for start, end, x, crop in windows(source, chunk_seconds, context_seconds):
            transformed = np.asarray(
                transform(x, start - crop.start) if with_position else transform(x),
                dtype="float64",
            )
            if len(transformed) != len(x):
                raise ValueError("Model changed duration in " + phase)
            if not fade:
                emit(transformed[crop])
            else:
                left = start
                if pending is not None:
                    blend_end = min(start + fade, end)
                    length = fade + blend_end - start
                    next_audio = transformed[
                        crop.start - fade : crop.start + blend_end - start
                    ]
                    ramp = np.arange(length) / max(1, 2 * fade - 1)
                    emit(pending[:length] * (1 - ramp) + next_audio * ramp)
                    left = blend_end
                right = end if end == total else max(left, end - fade)
                if right > left:
                    emit(
                        transformed[
                            crop.start + left - start : crop.start + right - start
                        ]
                    )
                pending = (
                    transformed[crop.stop - fade : crop.stop + fade].copy()
                    if end < total
                    else None
                )
            progress(phase, end / dsp.RATE)
    if count != total:
        raise ValueError("Chunk assembly lost or duplicated samples in " + phase)
    return {
        "samples": count,
        "rms": math.sqrt(energy / max(1, count)),
        "peak": peak,
        "estimated_true_peak_4x": meter.finish(),
    }

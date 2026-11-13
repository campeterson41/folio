"""Bounded-memory, multi-pass chapter processing with context at chunk boundaries."""

from __future__ import annotations
import hashlib, math
import numpy as np
import soundfile as sf
from collections import deque
from scipy import signal
from . import dsp, parallel


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


class Stem(sf.SoundFile):
    """A temporary stem. Closing it does not force the data to disk: stems are
    read back by the next stage and deleted, so syncing each one only adds
    writes. Published files are synced once, by their publisher."""

    def flush(self):
        pass


def sync(path):
    """Force a finished file to disk before it is published."""
    import os
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


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


def _true_peak_block(joined, a, b):
    high = signal.resample_poly(joined, 4, 1)
    return float(np.max(abs(high[a:b]), initial=0))


class TruePeakMeter:
    """4x peak estimate across block joins with real context, no boundary zero padding.

    Each block is oversampled with the tail of the previous one, so blocks are
    independent and can be measured by the window workers. The result is the
    maximum of the same per-block values as a serial pass.
    """

    def __init__(self):
        self.tail = np.empty(0)
        self.started = False
        self.peak = 0.0
        self.pending = deque()

    def _collect(self, limit):
        while len(self.pending) > limit:
            self.peak = max(self.peak, self.pending.popleft().result())

    def update(self, x):
        joined = np.concatenate((self.tail, x))
        if len(joined) <= 64:
            self.tail = joined.copy()
            return
        a = 128 if self.started else 0
        b = (len(joined) - 32) * 4
        if parallel.workers() > 1:
            self.pending.append(parallel.submit(_true_peak_block, joined, a, b))
            self._collect(parallel.workers())
        else:
            self.peak = max(self.peak, _true_peak_block(joined, a, b))
        self.tail = joined[-64:].copy()
        self.started = True

    def finish(self):
        self._collect(0)
        if len(self.tail):
            high = signal.resample_poly(self.tail, 4, 1)
            a = 128 if self.started else 0
            self.peak = max(self.peak, float(np.max(abs(high[a:]), initial=0)))
        return self.peak


def _transformed(transform, x, position):
    y = transform(x) if position is None else transform(x, position)
    extra = None
    if isinstance(y, tuple):
        y, extra = y
    return np.asarray(y, dtype="float64"), extra


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
    true_peak=True,
    workers=False,
    extras=None,
    durable=True,
):
    """Transform ``source`` window by window into ``target``.

    ``workers=True`` computes independent windows in the shared worker pool;
    the transform must then be a picklable function of the window alone. A
    transform may return ``(audio, extra)``; extras are appended to ``extras``
    in window order. ``true_peak=False`` skips the 4x peak estimate for stages
    whose callers only need RMS and sample peak. ``durable=False`` writes a
    temporary stem without syncing it to disk.
    """
    energy = 0.0
    peak = 0.0
    count = 0
    meter = TruePeakMeter() if true_peak else None
    pending = None
    total = sf.info(source).frames
    fade = round(crossfade_seconds * dsp.RATE)
    if fade > round(context_seconds * dsp.RATE):
        raise ValueError("Crossfade requires enough model context")
    with (sf.SoundFile if durable else Stem)(
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
            if meter is not None:
                meter.update(y)

        placed = deque()

        def tasks():
            for start, end, x, crop in windows(source, chunk_seconds, context_seconds):
                placed.append((start, end, crop, len(x)))
                yield transform, x, (start - crop.start if with_position else None)

        for transformed, extra in parallel.imap(_transformed, tasks(), enabled=workers):
            start, end, crop, length = placed.popleft()
            if extras is not None:
                extras.append(extra)
            if len(transformed) != length:
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
        "estimated_true_peak_4x": meter.finish() if meter is not None else None,
    }

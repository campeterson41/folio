"""Bounded tone matching and level control; no reference waveform synthesis.

A profile contains smooth frequency gains fitted on calibration speech only.
The reference is not needed to apply a saved profile to another recording.
"""

from __future__ import annotations
import math
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy import ndimage, signal

RATE = 48000
EPS = 1e-12


def load(path: str | Path, rate: int = RATE) -> np.ndarray:
    x, sr = sf.read(str(path), dtype="float64", always_2d=True)
    if not np.isfinite(x).all():
        raise ValueError(f"Non-finite samples: {path}")
    x = x.mean(axis=1)
    if sr != rate:
        g = math.gcd(sr, rate)
        x = signal.resample_poly(x, rate // g, sr // g)
    return x


def save(path: str | Path, x: np.ndarray, rate: int = RATE) -> None:
    if not np.isfinite(x).all() or np.max(np.abs(x), initial=0) >= 1:
        raise ValueError("Invalid or clipping output; no file written")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.stem + ".partial.wav")
    sf.write(str(part), x, rate, subtype="PCM_24")
    part.replace(target)


def db(value):
    return 20 * np.log10(np.maximum(value, EPS))


def rms(x):
    return float(np.sqrt(np.mean(np.square(x)))) if len(x) else 0.0


def frames(x, rate=RATE, window=0.04, hop=0.01):
    n = max(8, round(rate * window))
    h = max(1, round(rate * hop))
    padded = np.pad(x, (n // 2, n // 2 + h))
    return np.lib.stride_tricks.sliding_window_view(padded, n)[::h][
        : math.ceil(len(x) / h)
    ]


def envelope(x, rate=RATE, window=0.04, hop=0.01):
    n = max(1, round(rate * window))
    h = max(1, round(rate * hop))
    power = ndimage.uniform_filter1d(np.square(x), n, mode="constant")
    return np.sqrt(np.maximum(power[::h], 0.0) + EPS)


def speech_mask(raw, rate=RATE):
    return speech_mask_energy(envelope(raw, rate))


def speech_mask_energy(e):
    if not len(e):
        return np.zeros(0, dtype=bool)
    floor = np.percentile(e, 15)
    threshold = max(floor * 5.0, np.max(e, initial=0) * 0.035, 1e-7)
    # A constant tone needs a relative threshold, while a chapter with sparse
    # speech must not let its mostly quiet 85th percentile redefine room noise.
    if np.max(e, initial=0) < max(floor * 4.0, 1e-7):
        threshold = min(threshold, float(np.percentile(e, 85)) * 0.45)
    if np.max(e, initial=0) < 5e-6:
        return np.zeros(len(e), dtype=bool)
    return ndimage.maximum_filter1d((e > threshold).astype(np.uint8), size=5).astype(
        bool
    )


def smooth_spectrum(x, rate=RATE, mask=None):
    f, t, z = signal.stft(x, rate, nperseg=2048, noverlap=1568, boundary="zeros")
    if mask is None:
        good = np.ones(len(t), bool)
    else:
        good = mask[np.minimum((t / 0.01).astype(int), len(mask) - 1)]
    if np.count_nonzero(good) < 3:
        raise ValueError("Not enough calibration speech")
    p = np.mean(np.abs(z[:, good]) ** 2, axis=1)
    centers = np.geomspace(60.0, 18000.0, 48)
    band = []
    for c in centers:
        k = (f >= c / 2 ** (1 / 6)) & (f < c * 2 ** (1 / 6))
        if not k.any():
            k[np.argmin(abs(f - c))] = True
        band.append(np.mean(p[k]))
    return centers, 10 * np.log10(np.maximum(band, 1e-16))


def fit_tone(
    candidate,
    reference,
    raw,
    rate=RATE,
    interval=(5.0, 17.0),
    max_gain_db=6.0,
    smoothing=2.0,
):
    a, b = (round(v * rate) for v in interval)
    x = candidate[a:b]
    y = reference[a:b]
    r = raw[a:b]
    if min(len(x), len(y), len(r)) < rate:
        raise ValueError("Calibration interval is too short")
    mask = speech_mask(r, rate)
    c, px = smooth_spectrum(x, rate, mask)
    _, py = smooth_spectrum(y, rate, mask)
    difference = py - px
    # Remove overall gain: this stage changes tone, not the loudness target.
    mid = (c >= 150) & (c <= 8000)
    difference -= np.mean(difference[mid])
    difference = ndimage.gaussian_filter1d(difference, smoothing, mode="nearest")
    difference = np.clip(difference, -max_gain_db, max_gain_db)
    # Avoid learning boosts from inaudible bins outside the voice band.
    difference[c < 75] = min(0.0, difference[np.argmin(abs(c - 75))])
    return {
        "frequencies_hz": c.tolist(),
        "gains_db": difference.tolist(),
        "calibration_seconds": list(interval),
        "max_gain_db": max_gain_db,
        "method": "smoothed one-third-octave gain-neutral PSD ratio; no waveform reference at runtime",
    }


def apply_tone(x, profile, strength=1.0, rate=RATE):
    if strength == 0 or not profile:
        return np.array(x, copy=True)
    # A fixed FIR, rather than per-frame masks, prevents speech-dependent pumping.
    # Symmetric filter is latency-compensated by same-mode convolution.
    f = np.asarray(profile["frequencies_hz"])
    g = np.asarray(profile["gains_db"]) * strength
    n = 2049
    grid = np.linspace(0, rate / 2, 2049)
    gain = np.interp(np.log(np.maximum(grid, 1)), np.log(f), g)
    gain[0] = min(gain[0], 0.0)
    taps = signal.firwin2(
        n, grid / (rate / 2), 10 ** (gain / 20), window=("kaiser", 7.5)
    )
    return signal.fftconvolve(x, taps, mode="same")


def highpass(x, cutoff=65.0, rate=RATE):
    if cutoff <= 0:
        return x.copy()
    sos = signal.butter(2, cutoff, fs=rate, btype="highpass", output="sos")
    return signal.sosfiltfilt(sos, x)


def level_gain_curve(
    e, mask, strength=0.8, max_gain_db=12.0, anchor_db=None, stable_points=False
):
    if mask.sum() < 3:
        return np.zeros(len(e)), None
    if stable_points:
        # A 300ms RMS window straddling a word edge is quieter because it
        # includes a pause. Do not hold that edge's makeup gain for a long gap.
        stable = ndimage.binary_erosion(mask, structure=np.ones(31, dtype=bool))
        if stable.sum() >= 3:
            mask = stable
    anchor = float(np.median(db(e[mask]))) if anchor_db is None else float(anchor_db)
    gain = np.clip((anchor - db(e)) * strength, -max_gain_db, max_gain_db)
    indexes = np.flatnonzero(mask)
    gain = np.interp(np.arange(len(gain)), indexes, gain[indexes])
    return ndimage.gaussian_filter1d(gain, 35.0, mode="nearest"), anchor


def level(
    x, strength=0.8, target_db=-20.0, max_gain_db=12.0, rate=RATE, anchor_db=None
):
    """Slow leveler; gain decisions are made on speech and held through pauses."""
    e = envelope(x, rate, window=0.3, hop=0.01)
    mask = speech_mask(x, rate)
    gain, _ = level_gain_curve(e, mask, strength, max_gain_db, anchor_db)
    sample_gain = np.interp(np.arange(len(x)) / rate, np.arange(len(gain)) * 0.01, gain)
    return x * 10 ** (sample_gain / 20), {
        "leveler_gain_range_db": [
            float(gain.min(initial=0)),
            float(gain.max(initial=0)),
        ]
    }


def compress(
    x,
    ratio=1.5,
    threshold_db=-22.0,
    attack_ms=12.0,
    release_ms=180.0,
    knee_db=6.0,
    rate=RATE,
):
    if ratio <= 1:
        return x.copy(), {"compression_max_db": 0.0}
    # Audio is normalized near target before compressor so threshold is meaningful.
    power = ndimage.uniform_filter1d(x * x, max(1, round(rate * 0.01)), mode="nearest")
    level_db = db(np.sqrt(power + EPS))
    above = level_db - threshold_db
    k = knee_db / 2
    reduction = np.where(
        above < -k,
        0.0,
        np.where(
            above > k,
            (1 - 1 / ratio) * above,
            (1 - 1 / ratio) * (above + k) ** 2 / (2 * knee_db),
        ),
    )
    # IIR envelope smoothing at 1 ms keeps CPU bounded for long recordings.
    h = max(1, round(rate * 0.001))
    coarse = reduction[::h]
    a = math.exp(-h / (rate * attack_ms / 1000))
    b = math.exp(-h / (rate * release_ms / 1000))
    state = 0.0
    sm = np.empty_like(coarse)
    for i, value in enumerate(coarse):
        coef = a if value > state else b
        state = coef * state + (1 - coef) * value
        sm[i] = state
    full = np.interp(np.arange(len(x)), np.arange(len(sm)) * h, sm)
    return x * 10 ** (-full / 20), {"compression_max_db": float(sm.max(initial=0))}


def deess(x, strength=0.0, rate=RATE):
    if strength <= 0:
        return x.copy(), {"deess_max_db": 0.0}
    band = signal.sosfiltfilt(
        signal.butter(2, [4500, 10500], btype="bandpass", fs=rate, output="sos"), x
    )
    win = round(rate * 0.02)
    high = np.sqrt(ndimage.uniform_filter1d(band * band, win) + EPS)
    total = np.sqrt(ndimage.uniform_filter1d(x * x, win) + EPS)
    amount = np.clip((db(high / total) + 10) / 10, 0, 1) * strength
    h = max(1, round(rate * 0.001))
    coarse = ndimage.gaussian_filter1d(amount[::h], 8.0)
    amount = np.interp(np.arange(len(x)), np.arange(len(coarse)) * h, coarse)
    # Gentle fixed-band subtractive reduction, max 6 dB of band amplitude.
    return x - band * (1 - 10 ** (-amount * 6 / 20)), {
        "deess_max_db": float(amount.max(initial=0) * 6)
    }


def normalize(x, target_db=-20.0, peak_db=-4.0):
    peak = float(np.max(abs(x), initial=0))
    rr = rms(x)
    gain = (
        min(target_db - float(db(rr)), peak_db - float(db(peak))) if rr > 1e-7 else 0.0
    )
    return x * 10 ** (gain / 20), {
        "final_gain_db": gain,
        "rms_db": float(db(rr)) + gain,
        "sample_peak_db": float(db(peak)) + gain,
    }


def limit_peaks(x, peak_db=-4.0, lookahead_ms=5.0, release_ms=100.0, rate=RATE):
    """Look-ahead peak control with no audio delay, fixed clock and bounded gain."""
    if not len(x):
        return x.copy(), {"limiter_max_reduction_db": 0.0}
    h = max(1, round(rate * 0.001))
    padded = np.pad(np.abs(x), (0, (-len(x)) % h))
    peaks = padded.reshape(-1, h).max(axis=1)
    span = 2 * max(1, round(lookahead_ms)) + 1
    anticipated = ndimage.maximum_filter1d(peaks, span, mode="nearest")
    required = np.minimum(1.0, 10 ** (peak_db / 20) / np.maximum(anticipated, EPS))
    alpha = math.exp(-h / (rate * release_ms / 1000))
    state = 1.0
    gain = np.empty_like(required)
    for i, value in enumerate(required):
        state = value if value < state else alpha * state + (1 - alpha) * value
        gain[i] = state
    full = np.interp(np.arange(len(x)), np.arange(len(gain)) * h, gain)
    return x * full, {"limiter_max_reduction_db": float(-db(np.min(gain, initial=1.0)))}


def master(
    x,
    profile=None,
    tone_strength=1.0,
    level_strength=0.8,
    ratio=1.5,
    deess_strength=0.0,
    target_db=-20.0,
    peak_db=-4.0,
    cutoff=65.0,
    limiter=False,
):
    y = highpass(x, cutoff)
    y = apply_tone(y, profile, tone_strength)
    y, a = level(y, level_strength)
    y, _ = normalize(y, target_db, peak_db=0.0)
    y, b = compress(y, ratio=ratio)
    y, c = deess(y, deess_strength)
    lim = {"limiter_max_reduction_db": 0.0}
    if limiter and rms(y) > 1e-7:
        for _ in range(2):
            y = y * 10 ** ((target_db - float(db(rms(y)))) / 20)
            y, meter = limit_peaks(y, peak_db=peak_db - 0.1)
            lim["limiter_max_reduction_db"] += meter["limiter_max_reduction_db"]
    y, d = normalize(y, target_db, peak_db)
    # Report and honor a 4x oversampled estimate of the intersample peak.
    tp = float(np.max(abs(signal.resample_poly(y, 4, 1)), initial=0))
    safety = min(0.0, peak_db - float(db(tp)))
    y = y * 10 ** (safety / 20)
    d["true_peak_estimate_4x_db"] = float(db(tp)) + safety
    d["rms_db"] = float(db(rms(y)))
    d["true_peak_safety_gain_db"] = safety
    return y, {**a, **b, **c, **d, **lim, "sample_count": len(y)}

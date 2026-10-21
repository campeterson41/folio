"""CPU-only, source-based narration finishing; no models or resynthesis.

Every stage applies positive real gains, fixed filters or slowly varying
amplitude gain to the input recording itself. Noise is learned from quiet
regions of the same recording. No reference audio, transcript, network call or
neural model is used.

Pipeline (each stage is streamed in aligned 30 s chunks with 2 s context):

1. Preserved analysis branch: 12 dB-capped spectral noise reduction, 50 Hz
   high-pass, saved broad tone, saved gain and a -4 dBFS peak limiter. It is
   only analyzed to derive the level controls; it is not part of the output.
2. Quieter branch: stronger (1.6x, 18 dB-capped) noise reduction from the raw
   input with the same high-pass, tone, gain and peak limiter.
3. Level control: at most 2.5 dB of gentle compression and 9 dB of soft
   reduction between phrases, applied to the quieter branch.
4. Closer/warmer finish: up to 6 dB STFT attenuation where spectral energy is
   falling (a fixed late-tail heuristic, not room-impulse estimation) and a
   3 dB low shelf at 160 Hz.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import ndimage, signal

from . import dsp
from .adaptive import analyze, _read, HOP as ANALYSIS_HOP
from .streaming import stage

RATE = 48000
FFT = 2048
STEP = 512
# Aligned STFT grids across overlapping bounded blocks.
CHUNK_SECONDS = round(30*RATE/STEP)*STEP/RATE
CONTEXT_SECONDS = round(2*RATE/STEP)*STEP/RATE

HIGHPASS_HZ = 50
PEAK_DB = -4
PRESERVED_REDUCTION_DB = 12
QUIETER_SUBTRACTION = 1.6
QUIETER_REDUCTION_DB = 18
TAIL_HISTORY_FRAMES = 60
TAIL_DECAY_SECONDS = .12
TAIL_DELAY_FRAMES = 4
BASS_MIDPOINT_HZ = 160
MAXIMUM_TAIL_DB = 6
MAXIMUM_BASS_DB = 4


def reduce_noise(x, noise_power, maximum_db):
    """Original-phase spectral attenuation, capped at ``maximum_db``."""
    if not 0 <= maximum_db <= 18:
        raise ValueError('Invalid reduction limit')
    if maximum_db == 0:
        return x.copy()
    _, _, z = signal.stft(x, fs=RATE, nperseg=FFT, noverlap=FFT-STEP,
                          boundary='zeros', padded=True)
    # Positive real gains retain the source STFT phase. Broad smoothing and a
    # hard attenuation floor prevent narrow holes and hard spectral gates.
    p = ndimage.gaussian_filter(abs(z)**2, sigma=(2, 3))
    ratio = np.maximum(1-noise_power[:, None]/np.maximum(p, 1e-25), 0)
    gain_db = np.maximum(10*np.log10(np.maximum(ratio, 1e-12)), -maximum_db)
    gain_db = ndimage.gaussian_filter(gain_db, sigma=(2, 5))
    _, result = signal.istft(z*10**(gain_db/20), fs=RATE, nperseg=FFT,
                             noverlap=FFT-STEP, boundary=True)
    return result[:len(x)]


def clean_quieter(x, noise_power):
    """Stronger over-subtraction for the quieter branch, capped at 18 dB."""
    _, _, z = signal.stft(x, RATE, nperseg=FFT, noverlap=FFT-STEP,
                          boundary='zeros', padded=True)
    p = ndimage.gaussian_filter(abs(z)**2, sigma=(2, 3))
    ratio = np.maximum(1-QUIETER_SUBTRACTION*noise_power[:, None]/np.maximum(p, 1e-25), 0)
    gain = np.maximum(10*np.log10(np.maximum(ratio, 1e-12)), -QUIETER_REDUCTION_DB)
    gain = ndimage.gaussian_filter(gain, sigma=(2, 5))
    _, y = signal.istft(z*10**(gain/20), RATE, nperseg=FFT, noverlap=FFT-STEP,
                        boundary=True)
    return y[:len(x)]


def noise_profile(source, features):
    """Median noise power from up to 256 quiet, aperiodic source regions."""
    levels = features['level_dbfs']
    quiet = (levels < np.percentile(levels, 25)) & (features['periodicity'] < .3)
    quiet = ndimage.binary_erosion(quiet, structure=np.ones(15))
    indexes = np.flatnonzero(quiet)
    indexes = indexes[(indexes*ANALYSIS_HOP >= FFT*2) &
                      (indexes*ANALYSIS_HOP+FFT*2 < features['samples'])]
    if len(indexes) < 3:
        raise ValueError('No reliable quiet source regions for noise estimation. '
                         'The quietest quarter of the take needs several unbroken '
                         'pauses of 0.3 s or longer containing only room noise.')
    indexes = indexes[np.linspace(0, len(indexes)-1, min(256, len(indexes))).astype(int)]
    spectra = []
    with sf.SoundFile(source) as f:
        for i in indexes:
            x = _read(f, int(i)*ANALYSIS_HOP-FFT*2, FFT*4)
            _, _, z = signal.stft(x, RATE, nperseg=FFT, noverlap=FFT-STEP,
                                  boundary=None)
            spectra.append(np.median(abs(z)**2, axis=1))
    return ndimage.gaussian_filter1d(np.median(spectra, axis=0), 2), indexes.tolist()


def controls(raw, preserved):
    """Frame gains in dB from raw activity and preserved-branch levels."""
    level = raw['level_dbfs']
    wet = preserved['level_dbfs']
    threshold = max(float(np.percentile(level, 15))+18,
                    float(np.percentile(level, 90))-35)
    active = level > threshold
    if active.sum() < 10:
        raise ValueError('Not enough speech for finishing')
    # Every active frame has a full guard beyond the smoothing kernel. Quiet
    # reduction therefore cannot close on a detected syllable or consonant.
    held = ndimage.maximum_filter1d(active.astype(float), 35, mode='nearest')
    opening = ndimage.gaussian_filter1d(held, 4, mode='nearest', truncate=4)
    quiet = -9*(1-opening)
    threshold_compression = float(np.percentile(wet[active], 70))
    reduction = np.clip((wet-threshold_compression)*(1-1/1.5), 0, 2.5)
    reduction = ndimage.gaussian_filter1d(reduction, 4, mode='nearest', truncate=4)
    return quiet-reduction, {
        'activity_threshold_dbfs': threshold,
        'compression_threshold_dbfs': threshold_compression, 'ratio': 1.5,
        'maximum_compression_db': 2.5, 'maximum_quiet_reduction_db': 9,
        'activity_guard_seconds': .34, 'gain_smoothing_seconds': .08,
        'maximum_quiet_at_detected_speech_db': float(np.max(abs(quiet[active]))),
        'gain_min_max_db': [float((quiet-reduction).min()), float((quiet-reduction).max())]}


def tail_mask(z, maximum_db):
    """Positive STFT gains in [10**(-maximum_db/20), 1] on falling energy."""
    if not 0 <= maximum_db <= MAXIMUM_TAIL_DB:
        raise ValueError(f'Tail attenuation must be between 0 and {MAXIMUM_TAIL_DB} dB')
    if maximum_db == 0:
        return np.ones(z.shape)
    power = ndimage.gaussian_filter(abs(z)**2, (2, 1))
    # Only past frames >=43 ms old contribute. Finite 640 ms support makes
    # aligned bounded-window processing reproducible across chunk boundaries.
    kernel = np.exp(-np.arange(TAIL_HISTORY_FRAMES) * STEP / RATE / TAIL_DECAY_SECONDS)
    kernel /= kernel.sum()
    kernel = np.r_[np.zeros(TAIL_DELAY_FRAMES), kernel]
    previous = signal.lfilter(kernel, [1.], power, axis=1)
    # Steady vowels are protected: reduction grows only when the recent past
    # exceeds current energy. Rising attacks cannot trigger this reduction.
    falling = np.clip((previous - 1.15*power) / np.maximum(previous, 1e-24), 0, 1)
    frequencies = np.fft.rfftfreq(FFT, 1/RATE)
    weight = np.minimum(frequencies/180, 1.)[:, None]
    db = -maximum_db * ndimage.gaussian_filter(falling * weight, (2, 1.5))
    return 10**(db/20)


def closer(x, maximum_db):
    if maximum_db == 0:
        return x.copy()
    _, _, z = signal.stft(x, RATE, nperseg=FFT, noverlap=FFT-STEP,
                          boundary='zeros', padded=True)
    _, y = signal.istft(z*tail_mask(z, maximum_db), RATE, nperseg=FFT,
                        noverlap=FFT-STEP, boundary=True)
    return y[:len(x)]


def bass_taps(gain_db=3.):
    if not 0 <= gain_db <= MAXIMUM_BASS_DB:
        raise ValueError(f'Bass shelf must be between 0 and {MAXIMUM_BASS_DB} dB')
    f = np.linspace(0, RATE/2, 4097)
    # Broad low shelf, 160 Hz midpoint. Symmetric FIR compensated by cropping;
    # no time shift, pitch shift, or signal-dependent tonal processing.
    response = 10**((gain_db / (1+(f/BASS_MIDPOINT_HZ)**4))/20)
    return signal.firwin2(2049, f/(RATE/2), response, window=('kaiser', 7.5))


def closer_warm(x, maximum_db, gain_db=3.):
    return signal.fftconvolve(closer(x, maximum_db), bass_taps(gain_db), mode='same')


def validate_preset(preset):
    tone = preset.get('tone')
    if not isinstance(tone, dict):
        raise ValueError('Preset needs a tone profile')
    f = np.asarray(tone.get('frequencies_hz'), dtype=float)
    g = np.asarray(tone.get('gains_db'), dtype=float)
    if (f.ndim != 1 or f.shape != g.shape or len(f) < 2 or not np.isfinite(f).all()
            or not np.isfinite(g).all() or (f <= 0).any() or (np.diff(f) <= 0).any()
            or f[-1] > RATE/2 or abs(g).max() > 6):
        raise ValueError('Tone needs increasing positive frequencies and gains within ±6 dB')
    gain = preset.get('gain_db')
    if not isinstance(gain, (int, float)) or not math.isfinite(gain) or not -12 <= gain <= 12:
        raise ValueError('Preset gain_db must be between -12 and +12 dB')
    return tone, float(gain)


def _metrics_ok(metrics, samples, phase):
    if metrics['samples'] != samples:
        raise ValueError(f'{phase} changed the sample count')
    # PCM stages would quantize values at full scale; refuse instead.
    if metrics['peak'] >= 1:
        raise ValueError(f'{phase} would clip ({metrics["peak"]:.3f} peak)')


def render(source, workdir, preset, tail_db=6., bass_db=3., progress=None):
    """Render PCM-24 ``workdir/final.wav``; returns its path and details.

    ``workdir`` is an empty private directory owned by the caller, which is
    responsible for publication and for removing it on error or interruption.
    """
    source, tmp = Path(source), Path(workdir)
    progress = progress or (lambda *a: None)
    tone, gain_db = validate_preset(preset)
    if not 0 <= tail_db <= MAXIMUM_TAIL_DB:
        raise ValueError(f'Tail attenuation must be between 0 and {MAXIMUM_TAIL_DB} dB')
    if not 0 <= bass_db <= MAXIMUM_BASS_DB:
        raise ValueError(f'Bass shelf must be between 0 and {MAXIMUM_BASS_DB} dB')
    raw = analyze(source)
    samples = raw['samples']
    noise, noise_indexes = noise_profile(source, raw)
    quiet_level = float(np.median(raw['level_dbfs'][noise_indexes]))
    speech_level = float(np.percentile(raw['level_dbfs'], 90))
    if speech_level-quiet_level < 15:
        raise ValueError('Quiet regions are within 15 dB of speech; no reliable '
                         'noise reference. Record a little room tone between phrases.')
    gain = 10**(gain_db/20)
    chunk, context = CHUNK_SECONDS, CONTEXT_SECONDS
    reductions = {'preserved': [], 'quieter': []}

    def preserved_tone(x):
        y = reduce_noise(x, noise, PRESERVED_REDUCTION_DB)
        return dsp.apply_tone(dsp.highpass(y, HIGHPASS_HZ), tone)

    toned = tmp/'preserved-toned.wav'
    stage(source, toned, preserved_tone, progress, 'preserved branch',
          chunk, context, subtype='DOUBLE')

    def preserved_finish(x):
        y, meta = dsp.limit_peaks(x*gain, peak_db=PEAK_DB)
        reductions['preserved'].append(meta['limiter_max_reduction_db'])
        return y

    preserved = tmp/'preserved.wav'
    m = stage(toned, preserved, preserved_finish, progress, 'preserved peak protection',
              chunk, context, subtype='PCM_24')
    _metrics_ok(m, samples, 'Preserved branch')
    toned.unlink()
    gains, settings = controls(raw, analyze(preserved))
    preserved.unlink()

    def quieter_branch(x):
        y = dsp.apply_tone(dsp.highpass(clean_quieter(x, noise), HIGHPASS_HZ), tone)
        y, meta = dsp.limit_peaks(y*gain, peak_db=PEAK_DB)
        reductions['quieter'].append(meta['limiter_max_reduction_db'])
        return y

    quieter = tmp/'quieter.wav'
    stage(source, quieter, quieter_branch, progress, 'quieter branch',
          chunk, context, subtype='DOUBLE')

    def level_control(x, start):
        curve = np.interp((start+np.arange(len(x)))/ANALYSIS_HOP,
                          np.arange(len(gains)), gains)
        return x*10**(curve/20)

    controlled = tmp/'controlled.wav'
    m = stage(quieter, controlled, level_control, progress, 'level control',
              chunk, context, with_position=True, subtype='PCM_24')
    _metrics_ok(m, samples, 'Level control')
    quieter.unlink()

    final = tmp/'final.wav'
    metrics = stage(controlled, final, lambda x: closer_warm(x, tail_db, bass_db),
                    progress, 'closer-warm finish', chunk, context, subtype='PCM_24')
    _metrics_ok(metrics, samples, 'Closer-warm finish')
    if metrics['estimated_true_peak_4x'] >= .99:
        raise ValueError('Insufficient true-peak headroom after the bass shelf; '
                         'try a lower --bass-db')
    controlled.unlink()
    return final, {
        'samples': samples, 'metrics': metrics, 'gain_db': gain_db,
        'tone': {'frequencies_hz': list(tone['frequencies_hz']),
                 'gains_db': list(tone['gains_db'])},
        'noise_profile': {'source_frames': len(noise_indexes),
                          'median_quiet_level_dbfs': quiet_level,
                          'speech_level_p90_dbfs': speech_level},
        'level_controls': settings,
        'limiter_max_reduction_db': {k: max(v) for k, v in reductions.items()},
        'settings': {
            'sample_rate': RATE, 'stft_size': FFT, 'stft_hop': STEP,
            'chunk_samples': round(chunk*RATE), 'context_samples': round(context*RATE),
            'highpass_hz': HIGHPASS_HZ, 'peak_limit_dbfs': PEAK_DB,
            'preserved_branch_max_reduction_db': PRESERVED_REDUCTION_DB,
            'quieter_branch_subtraction': QUIETER_SUBTRACTION,
            'quieter_branch_max_reduction_db': QUIETER_REDUCTION_DB,
            'maximum_tail_attenuation_db': tail_db, 'bass_shelf_db': bass_db,
            'bass_shelf_midpoint_hz': BASS_MIDPOINT_HZ,
            'tail_history_seconds': TAIL_HISTORY_FRAMES*STEP/RATE,
            'tail_history_decay_seconds': TAIL_DECAY_SECONDS,
            'tail_prediction_delay_frames': TAIL_DELAY_FRAMES,
            'intermediate_subtypes': ['DOUBLE', 'PCM_24', 'DOUBLE', 'PCM_24', 'PCM_24']},
    }

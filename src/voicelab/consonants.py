"""Experimental source-fricative preservation after enhancement.

This is a separate optional experiment. It is not enabled in the preferred F
preset until listening establishes that it helps. No target audio is required.
"""
import numpy as np
from scipy import signal, ndimage
from . import dsp
from .consonant_plan import plan, source_profile, gain_curve


def preserve_fullband(source, enhanced, strength=1., delay_samples=2,
                     voiced_fricatives=False):
    """Automatically bypass suppressed fricatives with tone-matched source.

    No words, timestamps, external reference or clip-specific tone profile are
    accepted. Detection uses missing high-band energy above the recording's own
    noise floor, beside voiced speech. Whole-band replacement avoids combining
    a restored S with a still-damaged lower-frequency onset.
    """
    source = np.asarray(source, dtype=np.float64)
    enhanced = np.asarray(enhanced, dtype=np.float64)
    if source.ndim != 1 or source.shape != enhanced.shape or len(source) < 4800:
        raise ValueError('Expected matching mono audio of at least 100 ms')
    if not np.isfinite(source).all() or not np.isfinite(enhanced).all() or not 0 <= strength <= 1:
        raise ValueError('Invalid preservation inputs')
    rate, hop = 48000, 480
    if not np.any(source) or strength == 0:
        return enhanced.copy(), {'regions_seconds': [], 'strength': strength}
    low = signal.butter(3, [300, 3000], fs=rate, btype='bandpass', output='sos')
    high = signal.butter(3, [4500, 14000], fs=rate, btype='bandpass', output='sos')
    sl = dsp.envelope(signal.sosfiltfilt(low, source))
    wl = dsp.envelope(signal.sosfiltfilt(low, enhanced))
    sh = dsp.envelope(signal.sosfiltfilt(high, source))
    wh = dsp.envelope(signal.sosfiltfilt(high, enhanced))
    clipped = ndimage.maximum_filter1d((abs(source) >= .98).astype(np.uint8), 2401)[::hop] > 0
    regions,body,indexes = plan(sl,wl,sh,wh,clipped,len(source),voiced_fricatives,
                                lambda a,b:np.max(abs(source[a:b]),initial=0)<.98)
    if not regions:
        return enhanced.copy(), {'regions_seconds': [], 'strength': strength}
    profile = source_profile(body,regions,len(source),
                             lambda a,n:source[a:a+n],lambda a,n:enhanced[a:a+n])
    if profile is None:
        return enhanced.copy(), {'regions_seconds': [], 'strength': strength,
                                 'reason': 'Insufficient intact speech to match timbre'}
    dry = dsp.apply_tone(dsp.highpass(source, 50), profile)
    dl = dsp.envelope(signal.sosfiltfilt(low, dry))
    g = gain_curve(dl,wl,indexes)
    dry *= 10**(np.interp(np.arange(len(dry))/hop, np.arange(len(g)), g)/20)
    if delay_samples > 0: dry = np.pad(dry,(delay_samples,0))[:len(dry)]
    elif delay_samples < 0: dry = np.pad(dry[-delay_samples:],(0,-delay_samples))
    result = enhanced.copy()
    for a, b in regions:
        fade = min(1200, (b-a)//2)
        w = np.ones(b-a)*strength
        ramp = .5-.5*np.cos(np.linspace(0,np.pi,fade))
        w[:fade] *= ramp
        w[-fade:] *= ramp[::-1]
        result[a:b] += w*(dry[a:b]-enhanced[a:b])
    return result, {'regions_seconds': [[a/rate,b/rate] for a,b in regions],
                    'strength': strength, 'source_tone_profile': profile,
                    'voiced_fricatives': voiced_fricatives,
                    'detector': 'missing fricative energy, noise floor, neighboring voiced speech',
                    'note': 'Automatic full-band source preservation; listening evaluation required.'}


def preserve(source, enhanced, strength=.25, source_profile=None, delay_samples=2):
    source=np.asarray(source,dtype=np.float64)
    enhanced=np.asarray(enhanced,dtype=np.float64)
    if source.ndim!=1 or source.shape!=enhanced.shape or not len(source):
        raise ValueError('Source and enhanced audio must have matching nonempty mono shapes')
    if not np.isfinite(source).all() or not np.isfinite(enhanced).all() or not 0<=strength<=1:
        raise ValueError('Invalid samples or preservation strength')
    if not np.any(source) or strength==0:return enhanced.copy(),{'active_seconds':0.,'strength':strength}
    dry=dsp.apply_tone(source,source_profile) if source_profile else source.copy()
    low=signal.butter(3,[300,3000],fs=48000,btype='bandpass',output='sos')
    high=signal.butter(3,[4500,14000],fs=48000,btype='bandpass',output='sos')
    raw_low=signal.sosfiltfilt(low,dry);wet_low=signal.sosfiltfilt(low,enhanced)
    # A robust speech-body gain match, independent of the target reference.
    e=dsp.envelope(raw_low)
    mask=dsp.speech_mask_energy(e)
    if mask.sum()<3:return enhanced.copy(),{'active_seconds':0.,'strength':strength}
    raw_e=dsp.envelope(raw_low);wet_e=dsp.envelope(wet_low)
    gain=float(np.median(wet_e[mask]/np.maximum(raw_e[mask],1e-10)))
    dry*=gain
    if delay_samples>0:dry=np.pad(dry,(delay_samples,0))[:len(dry)]
    elif delay_samples<0:dry=np.pad(dry[-delay_samples:],(0,-delay_samples))
    dry_high=signal.sosfiltfilt(high,dry);wet_high=signal.sosfiltfilt(high,enhanced)
    def power(x):return np.maximum(ndimage.uniform_filter1d(x*x,1920,mode='constant'),1e-20)
    dh=power(dry_high);wh=power(wet_high);dl=power(signal.sosfiltfilt(low,dry))
    # Fricative-dominant frames, demonstrably above the quiet floor and with
    # missing high-band detail. Neither voiced bass nor steady hiss should pass.
    floor=max(float(np.percentile(dh[::480],10)),1e-18)
    ratio=10*np.log10(dh/dl);snr=10*np.log10(dh/floor);missing=10*np.log10(dh/wh)
    gate=(ratio>0)&(snr>15)&(missing>3)
    # Never reintroduce source clipping transients.
    clipped=ndimage.maximum_filter1d((abs(source)>=.98).astype(np.uint8),1921)>0
    gate &= ~clipped
    weights=ndimage.gaussian_filter1d(gate.astype(float),480)
    weights[weights<.01]=0
    result=enhanced+strength*weights*(dry_high-wet_high)
    if not np.isfinite(result).all():raise ValueError('Invalid preservation output')
    return result,{'strength':strength,'active_seconds':float(np.count_nonzero(weights)/48000),
                   'source_gain_db':float(dsp.db(gain)),'delay_samples':delay_samples,
                   'band_hz':[4500,14000],'note':'Experimental automatic fricative gate, not an accepted repair.'}

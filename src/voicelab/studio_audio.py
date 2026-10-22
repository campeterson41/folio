"""File-backed audio operations for the approved studio recipe.

Audio buffers are bounded; global level/gain decisions use coarse frame arrays.
The original recording and all pre-existing outputs are treated as read-only.
"""
from pathlib import Path
import math
import numpy as np
import soundfile as sf
from scipy import ndimage
from . import dsp
from .streaming import windows, TruePeakMeter

RATE = 48000
BLOCK = RATE * 30


class CheckedWriter:
    def __init__(self,path,frames,subtype):
        self.pcm=subtype.startswith('PCM_')
        self.path=Path(path)
        self.file=sf.SoundFile(path,'w',samplerate=RATE,channels=1,subtype=subtype,
            format='RF64' if frames*8>3900000000 else 'WAV')
    def write(self,x):
        if not np.isfinite(x).all():raise ValueError('Non-finite audio before write')
        if self.pcm and np.max(abs(x),initial=0)>=1:
            raise ValueError(f'PCM intermediate {self.path.name} would clip (peak {np.max(abs(x)):.6f}); nothing published')
        self.file.write(x)
    def __enter__(self):return self
    def __exit__(self,*exc):self.file.close()


def writer(path,frames,subtype='DOUBLE'):
    return CheckedWriter(path,frames,subtype)


def transform(source, target, function, context=.1, chunk=30, subtype='DOUBLE', position=False):
    info = sf.info(source)
    with writer(target, info.frames, subtype) as out:
        for start, end, x, crop in windows(source, chunk, context):
            y = function(x, start-crop.start) if position else function(x)
            if len(y) != len(x) or not np.isfinite(y).all():
                raise ValueError('Processing changed the clock or produced non-finite samples')
            out.write(y[crop])
    if sf.info(target).frames != info.frames:
        raise ValueError('Processing changed the sample count')
    return Path(target)


def envelopes(source, window=.04, hop=.01, mode='constant'):
    values = []
    n = max(1, round(RATE*window)); step = max(1, round(RATE*hop))
    for _, _, x, crop in windows(source, 30, max(.5, window*2)):
        power = ndimage.uniform_filter1d(x*x, n, mode=mode)
        e = np.sqrt(np.maximum(power[::step], 0)+dsp.EPS)
        values.append(e[crop.start//step:math.ceil(crop.stop/step)])
    return np.concatenate(values)


def mask(source):
    return dsp.speech_mask_energy(envelopes(source))


def stats(source, speech=None):
    energy=0.; count=0; peak=0.; meter=TruePeakMeter()
    with sf.SoundFile(source) as f:
        for start in range(0, f.frames, BLOCK):
            x=f.read(BLOCK)
            if not np.isfinite(x).all():
                raise ValueError('Non-finite audio')
            peak=max(peak,float(np.max(abs(x),initial=0)))
            meter.update(x)
            if speech is not None:
                indexes=np.minimum(np.arange(start,start+len(x))//480,len(speech)-1)
                x=x[speech[indexes]]
            energy+=float(np.sum(x*x)); count+=len(x)
    return dict(rms=math.sqrt(energy/max(1,count)), peak=peak,
                true_peak=meter.finish(), samples=sf.info(source).frames)


def gain(source, target, db, subtype='DOUBLE'):
    factor=10**(float(db)/20)
    return transform(source,target,lambda x:x*factor,context=0,subtype=subtype)


def normalize(source, target, speech, target_db=-23, subtype='DOUBLE'):
    rms=stats(source,speech)['rms']
    if rms<1e-10:
        raise ValueError('No usable speech for level matching')
    db=target_db-20*math.log10(rms)
    gain(source,target,db,subtype)
    return Path(target),db


def apply_curve(source, target, curve, hop=480, subtype='DOUBLE'):
    curve=np.asarray(curve,dtype=float)
    return transform(source,target,
        lambda x,start:x*10**(np.interp((start+np.arange(len(x)))/hop,
                                      np.arange(len(curve)),curve)/20),
        context=0,subtype=subtype,position=True)


def compress(source,target,ratio=3.5,threshold=-28,attack=3,release=140,knee=8,wet=.8):
    level=20*np.log10(np.maximum(envelopes(source,.01,.001,'nearest'),1e-12))
    above=level-threshold;k=knee/2
    reduction=np.where(above < -k,0,np.where(above > k,(1-1/ratio)*above,
        (1-1/ratio)*(above+k)**2/(2*knee)))
    a=math.exp(-.001/(attack/1000));b=math.exp(-.001/(release/1000))
    state=0.;sm=np.empty_like(reduction)
    for i,value in enumerate(reduction):
        coef=a if value>state else b;state=coef*state+(1-coef)*value;sm[i]=state
    transform(source,target,lambda x,start:x*((1-wet)+wet*10**(-np.interp(
        (start+np.arange(len(x)))/48,np.arange(len(sm)),sm)/20)),
        context=0,position=True)
    return dict(maximum_wet_reduction_db=float(sm.max(initial=0)),parallel_wet=wet)


def limit(source,target,peak_db=-9.841557099842,release=70,subtype='DOUBLE'):
    ceiling=10**(peak_db/20);peaks=[]
    with sf.SoundFile(source) as f:
        for x in f.blocks(BLOCK):
            padded=np.pad(abs(x),(0,(-len(x))%48))
            peaks.append(padded.reshape(-1,48).max(axis=1))
    peaks=np.concatenate(peaks)
    if np.max(peaks,initial=0)<=ceiling:
        gain(source,target,0,subtype);return {'maximum_reduction_db':0.}
    anticipated=ndimage.maximum_filter1d(peaks,11,mode='nearest')
    required=np.minimum(1,ceiling/np.maximum(anticipated,dsp.EPS))
    alpha=math.exp(-.001/(release/1000));state=1.;curve=np.empty_like(required)
    for i,value in enumerate(required):
        state=value if value<state else alpha*state+(1-alpha)*value;curve[i]=state
    # Interpolate linear gains, as in the original limiter (not dB).
    transform(source,target,lambda x,start:x*np.interp((start+np.arange(len(x)))/48,
        np.arange(len(curve)),curve),context=0,subtype=subtype,position=True)
    return {'maximum_reduction_db':float(-20*np.log10(np.min(curve,initial=1)))}


def add(source,other,target,other_gain=1.,source_gain=1.,subtype='DOUBLE'):
    frames=sf.info(source).frames
    if sf.info(other).frames!=frames:
        raise ValueError('Cannot combine different sample clocks')
    with sf.SoundFile(source) as a,sf.SoundFile(other) as b,writer(target,frames,subtype) as out:
        for x in a.blocks(BLOCK):
            y=x*source_gain+b.read(len(x))*other_gain
            if not np.isfinite(y).all():raise ValueError('Non-finite blend')
            out.write(y)
    return Path(target)


def validate_edits(edits,frames):
    """A user-supplied keep list, in source samples; never infer repeat cuts."""
    if edits is None:return [(0,frames)],0,0
    if not isinstance(edits,dict) or set(edits)-{'sample_rate','keep','crossfade_samples','edge_fade_samples'}:
        raise ValueError('Unsupported edit-list fields')
    if edits.get('sample_rate',RATE)!=RATE:raise ValueError('Edit-list rate must be48000')
    keep=edits.get('keep');fade=edits.get('crossfade_samples',0);edge=edits.get('edge_fade_samples',0)
    if (not isinstance(keep,list) or not keep or len(keep)>100000 or
        type(fade) is not int or type(edge) is not int or not 0<=fade<=4800 or not 0<=edge<=4800):
        raise ValueError('Invalid keep list or fades')
    last=0;result=[]
    for pair in keep:
        if (not isinstance(pair,list) or len(pair)!=2 or any(type(x)is not int for x in pair)
            or not 0<=pair[0]<pair[1]<=frames or pair[0]<last or pair[1]-pair[0]<max(1,fade*2)):
            raise ValueError('Keep intervals must be ordered, nonoverlapping and within the input')
        result.append(tuple(pair));last=pair[1]
    length=sum(b-a for a,b in result)-fade*(len(result)-1)
    if length<max(4800,edge*2):raise ValueError('Edited output is too short')
    return result,fade,edge


def cut(source,target,edits=None,subtype='DOUBLE'):
    info=sf.info(source);keep,fade,edge=validate_edits(edits,info.frames)
    frames=sum(b-a for a,b in keep)-fade*(len(keep)-1);mapping=[];position=0
    # Only a fade tail plus a disk block is held, even for long intervals.
    with sf.SoundFile(source) as f,writer(target,frames,subtype) as out:
        pending=np.empty(0)
        for i,(a,b) in enumerate(keep):
            start=position
            mapping.append(dict(source_start_sample=a,source_end_sample=b,
                                output_start_sample=start,output_end_sample=start+b-a))
            f.seek(a);remaining=b-a
            while remaining:
                x=f.read(min(BLOCK,remaining));remaining-=len(x)
                if i and f.tell()-len(x)==a and fade:
                    ramp=np.linspace(0,1,fade)
                    pending[-fade:]=pending[-fade:]*(1-ramp)+x[:fade]*ramp
                    x=x[fade:]
                joined=np.concatenate((pending,x))
                retain=fade if remaining==0 and i<len(keep)-1 else 0
                emitted=joined[:-retain] if retain else joined
                if edge:
                    indexes=np.arange(position,position+len(emitted))
                    g=np.minimum(np.clip(indexes/max(1,edge-1),0,1),
                                 np.clip((frames-1-indexes)/max(1,edge-1),0,1))
                    emitted=emitted*g
                out.write(emitted);position+=len(emitted)
                pending=joined[-retain:] if retain else np.empty(0)
        if pending.size:raise ValueError('Edit assembly retained unexpected samples')
    if position!=frames or sf.info(target).frames!=frames:raise ValueError('Edit clock mismatch')
    return mapping


def robust_speech_mask(source):
    """Percentile-based speech activity, resistant to a single loud transient.

    Uses the preview analyzer's 80 Hz high-pass, 20 ms power, 10 ms hop,
    percentile noise floor, 30 ms opening and 20 ms hangover. No quiet timestamps.
    """
    from scipy import signal
    sos=signal.butter(4,80,fs=RATE,btype='highpass',output='sos')
    parts=[]
    for _,_,x,crop in windows(source,30,4):
        hp=signal.sosfiltfilt(sos,x)
        power=ndimage.uniform_filter1d(hp*hp,960,mode='constant')[::480]
        parts.append(power[crop.start//480:math.ceil(crop.stop/480)])
    power=np.concatenate(parts);level=10*np.log10(np.maximum(power,1e-16))
    floor=float(np.percentile(level,10));loud=float(np.percentile(level,95))
    threshold=min(max(floor+12,loud-40),loud-6)
    speech=level>threshold
    speech=ndimage.binary_opening(speech,structure=np.ones(3,bool))
    return ndimage.binary_dilation(speech,structure=np.ones(5,bool))


def preview_gain(source,warm,target=-22):
    speech=robust_speech_mask(source)
    raw_power=np.maximum(envelopes(source,.02)**2-dsp.EPS,0)
    warm_power=np.maximum(envelopes(warm,.02)**2-dsp.EPS,0)
    if speech.sum()<10:raise ValueError('Not enough speech for preview matching')
    lo,hi=np.percentile(raw_power[speech],[5,95])
    selected=speech&(raw_power>=lo)&(raw_power<=hi)
    return min(target-10*math.log10(max(float(np.mean(warm_power[selected])),1e-16)),36.)


def activity_mask(source):
    original=mask(source);robust=robust_speech_mask(source)
    fraction=float(np.mean(original));reference=float(np.mean(robust))
    fallback=fraction<.1 or fraction<.5*reference
    return (robust if fallback else original),dict(
        speech_mask_method='robust-percentile-fallback' if fallback else 'original-energy',
        original_speech_fraction=fraction,robust_speech_fraction=reference)

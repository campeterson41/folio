"""Bounded audio buffers with global consonant decisions for long recordings."""
import math
from pathlib import Path
import shutil
import tempfile

import numpy as np
import soundfile as sf
from scipy import signal,ndimage
from . import dsp
from .consonant_plan import plan,source_profile,gain_curve
from .streaming import windows,stage


def preserve_file(source,enhanced,target,chunk_seconds=30,voiced_fricatives=False,
                  progress=lambda phase,seconds:None):
    source,enhanced,target=map(Path,(source,enhanced,target))
    if target.resolve() in [source.resolve(),enhanced.resolve()]:
        raise ValueError('Cannot overwrite an input')
    if target.exists():raise FileExistsError(target)
    if chunk_seconds<1 or round(chunk_seconds*100)!=chunk_seconds*100:
        raise ValueError('Use chunks on the 10ms grid, at least one second')
    info,wet_info=sf.info(source),sf.info(enhanced)
    if (info.frames!=wet_info.frames or info.frames<4800 or
        info.samplerate!=48000 or wet_info.samplerate!=48000 or
        info.channels!=1 or wet_info.channels!=1):
        raise ValueError('Expected matching 48 kHz mono recordings')
    low=signal.butter(3,[300,3000],fs=48000,btype='bandpass',output='sos')
    high=signal.butter(3,[4500,14000],fs=48000,btype='bandpass',output='sos')
    def features(path,both=True):
        lows=[];highs=[];clips=[]
        for start,end,x,crop in windows(path,chunk_seconds,4):
            if not np.isfinite(x).all():raise ValueError('Non-finite input audio')
            c=slice(crop.start//480,math.ceil(crop.stop/480))
            lows.append(dsp.envelope(signal.sosfiltfilt(low,x))[c])
            if both:
                highs.append(dsp.envelope(signal.sosfiltfilt(high,x))[c])
                clips.append((ndimage.maximum_filter1d((abs(x)>=.98).astype(np.uint8),2401)[::480]>0)[c])
            progress('consonant analysis',end/48000)
        if not both:return np.concatenate(lows)
        return np.concatenate(lows),np.concatenate(highs),np.concatenate(clips)
    sl,sh,clipped=features(source)
    wl,wh,_=features(enhanced)
    with sf.SoundFile(source) as dry_reader,sf.SoundFile(enhanced) as wet_reader:
        def read(f,a,n):
            f.seek(a)
            return f.read(n,dtype='float64')
        regions,body,indexes=plan(sl,wl,sh,wh,clipped,info.frames,voiced_fricatives,
                                  lambda a,b:np.max(abs(read(dry_reader,a,b-a)),initial=0)<.98)
        profile=source_profile(body,regions,info.frames,
            lambda a,n:read(dry_reader,a,n),lambda a,n:read(wet_reader,a,n)) if regions else None
    target.parent.mkdir(parents=True,exist_ok=True)
    result={'regions_seconds':[[a/48000,b/48000] for a,b in regions],
            'voiced_fricatives':voiced_fricatives,'strength':1.,
            'source_tone_profile':profile,'chunk_seconds':chunk_seconds,
            'maximum_audio_window_seconds':chunk_seconds+8,
            'detector':'missing fricative energy, noise floor, neighboring voiced speech',
            'note':'File-backed automatic preservation; global decisions, no word locations.'}
    if profile is None:
        result['regions_seconds']=[]
        shutil.copyfile(enhanced,target)
        return result
    with tempfile.TemporaryDirectory(prefix='voice-lab-consonants-',dir=target.parent) as tmp:
        dry_path=Path(tmp)/'tone-matched-source.wav'
        stage(source,dry_path,lambda x:dsp.apply_tone(dsp.highpass(x,50),profile),
              progress,'source timbre',chunk_seconds,4,subtype='DOUBLE')
        dl=features(dry_path,both=False)
        gains=gain_curve(dl,wl,indexes)
        fmt='RF64' if info.frames*3>3900000000 else 'WAV'
        with sf.SoundFile(enhanced) as wet,sf.SoundFile(target,'w',samplerate=48000,
                channels=1,subtype='PCM_24',format=fmt) as out:
            region_index=0
            for start,end,x,crop in windows(dry_path,chunk_seconds,4):
                position=start-crop.start
                g=np.interp((np.arange(len(x))+position)/480,np.arange(len(gains)),gains)
                x=x*10**(g/20)
                # TF's fixed two-sample alignment, with context retained across
                # every disk block. No new boundary or time warp is introduced.
                x=np.pad(x,(2,0))[:len(x)]
                dry=x[crop]
                wet.seek(start);y=wet.read(end-start,dtype='float64')
                while region_index<len(regions) and regions[region_index][1]<=start:
                    region_index+=1
                for a,b in regions[region_index:]:
                    if a>=end:break
                    lo,hi=max(a,start),min(b,end)
                    positions=np.arange(lo,hi)
                    fade=min(1200,(b-a)//2)
                    weight=np.ones(hi-lo)
                    left=positions<a+fade;right=positions>=b-fade
                    weight[left]=.5-.5*np.cos(np.pi*(positions[left]-a)/(fade-1))
                    weight[right]=.5-.5*np.cos(np.pi*(b-1-positions[right])/(fade-1))
                    c=slice(lo-start,hi-start)
                    y[c]+=weight*(dry[c]-y[c])
                if not np.isfinite(y).all() or np.max(abs(y),initial=0)>=1:
                    raise ValueError('Invalid preservation output')
                out.write(y)
                progress('preserve consonants',end/48000)
    if sf.info(target).frames!=info.frames:raise RuntimeError('Preservation changed duration')
    return result

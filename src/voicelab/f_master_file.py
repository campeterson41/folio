"""File-backed F mastering, retaining the approved global level decisions."""
import math
from pathlib import Path
import tempfile

import numpy as np
import soundfile as sf
from scipy import signal
from . import dsp
from .streaming import stage, windows


def master_file(source, target, preset, chunk_seconds=30,
                progress=lambda phase,seconds: None):
    source,target=Path(source),Path(target)
    if source.resolve()==target.resolve():raise ValueError('Cannot overwrite source')
    if target.exists():raise FileExistsError(target)
    if chunk_seconds<1 or round(chunk_seconds*100)!=chunk_seconds*100:
        raise ValueError('Use chunks of at least one second on the 10ms grid')
    info=sf.info(source)
    if info.samplerate!=48000 or info.channels!=1:
        raise ValueError('Expected 48 kHz mono restoration')
    settings=preset['mastering_settings'];profile=preset['tone_profile']
    if settings.get('limiter',False) or settings.get('deess_strength',0)!=0:
        raise ValueError('This file path implements the frozen F recipe without limiter or de-esser')
    target.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='voice-lab-f-master-',dir=target.parent) as tmp:
        tmp=Path(tmp)
        toned=tmp/'toned.wav'
        def tone(x):
            return dsp.apply_tone(dsp.highpass(x,settings.get('cutoff',50)),
                                  profile,settings.get('tone_strength',1))
        def run(src,dst,fn,phase,**kwargs):
            return stage(src,dst,fn,progress,phase,chunk_seconds,
                         context_seconds=4,subtype='DOUBLE',**kwargs)
        run(source,toned,tone,'tone')
        levels=[];voices=[]
        for _,_,x,crop in windows(toned,chunk_seconds,4):
            c=slice(crop.start//480,math.ceil(crop.stop/480))
            levels.append(dsp.envelope(x,window=.3)[c])
            voices.append(dsp.envelope(x)[c])
        global_gain,anchor=dsp.level_gain_curve(np.concatenate(levels),
            dsp.speech_mask_energy(np.concatenate(voices)),
            strength=settings.get('level_strength',.4))
        leveled=tmp/'leveled.wav'
        def level(x,position):
            g=np.interp((np.arange(len(x))+position)/48000,
                        np.arange(len(global_gain))*.01,global_gain)
            return x*10**(g/20)
        stats=run(toned,leveled,level,'level',with_position=True)
        target_db=settings.get('target_db',-20)
        peak_db=settings.get('peak_db',-4)
        pre_gain=min(target_db-float(dsp.db(stats['rms'])),
                     -float(dsp.db(stats['peak']))) if stats['rms']>1e-7 else 0.
        compressed=tmp/'compressed.wav'
        stats=run(leveled,compressed,
            lambda x:dsp.compress(x*10**(pre_gain/20),ratio=settings.get('ratio',1.3))[0],
            'compress')
        gain=min(target_db-float(dsp.db(stats['rms'])),
                 peak_db-float(dsp.db(max(stats['peak'],stats['estimated_true_peak_4x'])))) if stats['rms']>1e-7 else 0.
        intermediate=tmp/'f-pcm24.wav'
        fmt='RF64' if info.frames*3>3900000000 else 'WAV'
        with sf.SoundFile(intermediate,'w',samplerate=48000,channels=1,
                          subtype='PCM_24',format=fmt) as out:
            for _,_,x,_ in windows(compressed,chunk_seconds,0):
                out.write(x*10**(gain/20))
        eq=preset['preferred_final_eq']
        if eq['type']!='peaking_biquad':raise ValueError('Unsupported final EQ')
        A=10**(eq['gain_db']/40);w=2*np.pi*eq['hz']/48000;alpha=np.sin(w)/(2*eq['q'])
        b=np.array([1+alpha*A,-2*np.cos(w),1-alpha*A])
        a=np.array([1+alpha/A,-2*np.cos(w),1-alpha/A]);b,a=b/a[0],a/a[0]
        state=np.zeros(2)
        with sf.SoundFile(target,'w',samplerate=48000,channels=1,
                          subtype='PCM_24',format=fmt) as out:
            for _,end,x,_ in windows(intermediate,chunk_seconds,0):
                y,state=signal.lfilter(b,a,x,zi=state)
                if not np.isfinite(y).all() or np.max(abs(y),initial=0)>=1:
                    raise ValueError('Invalid final mastered samples')
                out.write(y)
                progress('presence EQ',end/48000)
        if sf.info(target).frames!=info.frames:raise RuntimeError('Mastering changed sample count')
    return {'samples':info.frames,'global_level_anchor_db':anchor,'pre_gain_db':pre_gain,
            'final_gain_db':gain,'chunk_seconds':chunk_seconds,
            'maximum_audio_window_seconds':chunk_seconds+8,
            'note':'Global F level decisions; DOUBLE intermediates; PCM24 before stateful presence EQ.'}

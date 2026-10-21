"""Optional final loudness/peak control; no EQ or neural restoration changes."""
from pathlib import Path
import shutil
import tempfile

import numpy as np
import soundfile as sf
from . import dsp
from .streaming import stage,windows,TruePeakMeter


def measure(path,chunk_seconds=30):
    energy=0.;peak=0.;count=0;meter=TruePeakMeter()
    for _,_,x,_ in windows(path,chunk_seconds,0):
        if not np.isfinite(x).all():raise ValueError('Non-finite audio')
        energy+=float(np.dot(x,x));peak=max(peak,float(np.max(abs(x),initial=0)))
        count+=len(x);meter.update(x)
    return {'samples':count,'rms':float(np.sqrt(energy/max(1,count))),
            'peak':peak,'true_peak':meter.finish()}


def finish_file(source,target,target_db=-20.,peak_db=-4.,maximum_boost_db=24.,
                chunk_seconds=30,progress=lambda phase,seconds:None):
    source,target=Path(source),Path(target)
    if source.resolve()==target.resolve():raise ValueError('Cannot overwrite input')
    if target.exists():raise FileExistsError(target)
    if chunk_seconds<1:raise ValueError('Chunks must be at least one second')
    if not -40<=target_db<=-12 or not -12<=peak_db<=-1 or not 0<=maximum_boost_db<=30:
        raise ValueError('Invalid finishing levels')
    info=sf.info(source)
    if info.samplerate!=48000 or info.channels!=1:
        raise ValueError('Expected 48 kHz mono audio')
    stats=measure(source,chunk_seconds);initial=dict(stats)
    target.parent.mkdir(parents=True,exist_ok=True)
    if stats['rms']<=1e-7:
        shutil.copyfile(source,target)
        return {'bypassed':'silence or near-silence','samples':info.frames}
    total_boost=0.;reductions=[]
    with tempfile.TemporaryDirectory(prefix='voice-lab-levels-',dir=target.parent) as tmp:
        current=source
        for i in range(2):
            boost=min(target_db-float(dsp.db(stats['rms'])),maximum_boost_db-total_boost)
            reduction=[0.]
            def transform(x):
                y,meter=dsp.limit_peaks(x*10**(boost/20),peak_db=peak_db-.1)
                reduction[0]=max(reduction[0],meter['limiter_max_reduction_db'])
                return y
            next_path=Path(tmp)/f'limited-{i}.wav'
            s=stage(current,next_path,transform,progress,'peak control',chunk_seconds,
                    context_seconds=4,subtype='DOUBLE')
            stats={'rms':s['rms'],'peak':s['peak'],'true_peak':s['estimated_true_peak_4x']}
            total_boost+=boost;reductions.append(reduction[0]);current=next_path
        gain=min(target_db-float(dsp.db(stats['rms'])),peak_db-float(dsp.db(stats['true_peak'])),
                 maximum_boost_db-total_boost)
        fmt='RF64' if info.frames*3>3900000000 else 'WAV'
        with sf.SoundFile(target,'w',samplerate=48000,channels=1,subtype='PCM_24',format=fmt) as out:
            for _,end,x,_ in windows(current,chunk_seconds,0):
                y=x*10**(gain/20)
                if not np.isfinite(y).all() or np.max(abs(y),initial=0)>=1:
                    raise ValueError('Invalid final samples')
                out.write(y);progress('level export',end/48000)
    final=measure(target,chunk_seconds)
    if final['samples']!=info.frames:raise RuntimeError('Level finishing changed duration')
    return {'samples':info.frames,'target_rms_dbfs':target_db,'peak_ceiling_dbfs':peak_db,
            'initial_rms_dbfs':float(dsp.db(initial['rms'])),
            'final_rms_dbfs':float(dsp.db(final['rms'])),
            'final_true_peak_dbfs':float(dsp.db(final['true_peak'])),
            'total_uniform_boost_db':total_boost+gain,
            'limiter_max_reduction_each_pass_db':reductions,
            'note':'Optional amplitude-only finishing after automatic preservation. '
                   'No EQ, speech synthesis, trimming or reference audio.'}

"""Experimental local, recording-aware performance protection.

No transcripts, word locations, cloud calls or reference audio enter decisions.
Pitch is a routing hint, not a diagnosis of damaged speech.
"""
from pathlib import Path
import tempfile
import numpy as np
import soundfile as sf
from scipy import signal,ndimage
from . import dsp
from .streaming import windows,stage
from .consonant_plan import source_profile,gain_curve

HOP=960

def analyze(path,batch_frames=512):
    """40ms windows on a global20ms grid; bounded input/FFT batches."""
    info=sf.info(path)
    if info.samplerate!=48000 or info.channels!=1:raise ValueError('Expected48kHz mono')
    levels=[];pitches=[];confidence=[];tilts=[]
    count=(info.frames+HOP-1)//HOP
    with sf.SoundFile(path) as f:
        for first in range(0,count,batch_frames):
            frames=min(batch_frames,count-first);start=first*HOP
            a=max(0,start-1920);b=min(info.frames,start+(frames-1)*HOP+1920+1920)
            f.seek(a);x=f.read(b-a,dtype='float64')
            if not np.isfinite(x).all():raise ValueError('Non-finite input')
            x=signal.resample_poly(x,1,3);offset=(start-a)//3
            needed=offset+(frames-1)*320+640
            x=np.pad(x,(0,max(0,needed-len(x))))
            blocks=np.lib.stride_tricks.sliding_window_view(x[offset:],640)[::320][:frames].copy()
            blocks-=blocks.mean(axis=1,keepdims=True)
            levels.append(20*np.log10(np.maximum(np.sqrt(np.mean(blocks**2,axis=1)),1e-12)))
            blocks*=np.hanning(640)
            spec=np.fft.rfft(blocks,2048,axis=1);power=abs(spec)**2
            freq=np.fft.rfftfreq(2048,1/16000)
            tilts.append(10*np.log10(np.maximum(power[:,freq>4500].sum(axis=1),1e-20)/
                np.maximum(power[:,(freq>300)&(freq<3000)].sum(axis=1),1e-20)))
            ac=np.fft.irfft(power,2048,axis=1)[:,:230]
            ac/=np.maximum(ac[:,:1],1e-20)
            region=ac[:,32:229];local=(region[:,1:-1]>region[:,:-2])&(region[:,1:-1]>=region[:,2:])
            peaks=np.where(np.pad(local,((0,0),(1,1))),region,-1.)
            best=peaks.max(axis=1)
            # Earliest strong periodic peak avoids selecting a longer subharmonic.
            lag=32+np.argmax(peaks>=np.maximum(best[:,None]*.9,.1),axis=1)
            pitches.append(16000/lag);confidence.append(ac[np.arange(frames),lag])
    return {'level_dbfs':np.concatenate(levels),'pitch_hz':np.concatenate(pitches),
            'periodicity':np.concatenate(confidence),'high_to_mid_db':np.concatenate(tilts),
            'samples':info.frames,'hop_seconds':.02}


def controller(features):
    level,pitch,confidence=[features[k] for k in ('level_dbfs','pitch_hz','periodicity')]
    floor=float(np.percentile(level,15));ceiling=float(np.percentile(level,90))
    threshold=max(floor+12,ceiling-35)
    if ceiling-floor<12:threshold=ceiling-25
    voiced=(level>threshold)&(confidence>.5)
    baseline=float(np.percentile(pitch[voiced],40)) if voiced.sum()>=10 else None
    selected=np.zeros(len(level),dtype=bool)
    if baseline:
        high=voiced&(pitch>baseline*1.35)
        high=ndimage.binary_opening(high,structure=np.ones(3))
        speech=ndimage.binary_closing(level>threshold,structure=np.ones(9))
        phrase_labels,_=ndimage.label(speech)
        for obj in ndimage.find_objects(phrase_labels):
            s=obj[0]
            if high[s].sum()>=3 and high[s].sum()>=.2*max(1,voiced[s].sum()):
                selected[s]=True
            elif high[s].sum()>=3:
                selected[s]=high[s]
        # Expand phrase edges to protect unvoiced onsets/ends.
        selected=ndimage.maximum_filter1d(selected.astype(np.uint8),13)>0
        # Leave unrelated long pauses with the restoration path.
        nearby=ndimage.maximum_filter1d((level>threshold).astype(np.uint8),15)>0
        selected&=nearby
        # A near-mic overload is a different condition from a register change.
        selected&=level<ceiling+12
    labels,_=ndimage.label(selected);regions=[]
    for obj in ndimage.find_objects(labels):
        s=obj[0];regions.append([s.start*HOP,min(features['samples'],s.stop*HOP)])
    return {'baseline_pitch_hz':baseline,'speech_threshold_dbfs':threshold,
            'voiced_frames':voiced,'selected':selected,'regions':regions,
            'overload_frames':level>=ceiling+12}


def _read(f,start,count):
    f.seek(max(0,start));x=f.read(max(0,min(count+min(start,0),len(f)-max(0,start))),dtype='float64')
    return np.pad(x,(max(0,-start),max(0,count-len(x)-max(0,-start))))[:count]


def protect(source,enhanced,conservative,target,chunk_seconds=30):
    source,enhanced,conservative,target=map(Path,(source,enhanced,conservative,target))
    if target.exists():raise FileExistsError(target)
    if target.resolve() in [p.resolve() for p in (source,enhanced,conservative)]:raise ValueError('Cannot overwrite input')
    infos=[sf.info(p) for p in (source,enhanced,conservative)]
    if any(i.frames!=infos[0].frames or i.samplerate!=48000 or i.channels!=1 for i in infos):
        raise ValueError('Expected matching48kHz mono inputs')
    features=analyze(source);decision=controller(features)
    wet_features=analyze(enhanced)
    trusted=decision['voiced_frames']&~decision['selected']&~decision['overload_frames']
    profile=None
    with sf.SoundFile(conservative) as dry,sf.SoundFile(enhanced) as wet:
        body=np.repeat(trusted,2)
        if trusted.sum()>=10:
            profile=source_profile(body,decision['regions'],features['samples'],
                lambda a,n:_read(dry,a,n),lambda a,n:_read(wet,a,n))
    if profile:
        # Match broad recording coloration gently, retaining character formants.
        profile['gains_db']=np.clip(np.asarray(profile['gains_db'])*.5,-3,3).tolist()
    target.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='folio-adaptive-',dir=target.parent) as tmp:
        tone=Path(tmp)/'gentle-tone.wav'
        stage(conservative,tone,lambda x:dsp.apply_tone(dsp.highpass(x,50),profile) if profile else dsp.highpass(x,50),
              lambda *a:None,'adaptive tone',chunk_seconds,4,subtype='DOUBLE')
        gentle_features=analyze(tone)
        indexes=np.flatnonzero(trusted)
        # Recording-derived level matching; high-register gaps interpolate from intact speech.
        if len(indexes)>=3:
            gains=gain_curve(10**(gentle_features['level_dbfs']/20),10**(wet_features['level_dbfs']/20),indexes)
        else:
            gains=np.zeros(len(features['level_dbfs']))
        # Estimate one alignment from intact voiced windows, with a strict5ms bound.
        lags=[]
        with sf.SoundFile(tone) as dry,sf.SoundFile(enhanced) as wet:
            for i in indexes[np.linspace(0,max(0,len(indexes)-1),min(24,len(indexes))).astype(int)]:
                a=int(i)*HOP;n=4096
                if a+n>infos[0].frames:continue
                x=_read(dry,a,n);y=_read(wet,a,n)
                x-=x.mean();y-=y.mean()
                cor=signal.correlate(y,x,mode='full',method='fft');mid=n-1
                lag=int(np.argmax(cor[mid-240:mid+241])-240)
                if abs(lag)<240:lags.append(lag)
        delay=int(round(np.median(lags))) if lags else 0
        with sf.SoundFile(tone) as dry,sf.SoundFile(enhanced) as wet,sf.SoundFile(target,'x',
                samplerate=48000,channels=1,subtype='PCM_24',format='RF64') as out:
            for start in range(0,infos[0].frames,round(chunk_seconds*48000)):
                n=min(round(chunk_seconds*48000),infos[0].frames-start)
                x=_read(dry,start-delay,n)
                g=np.interp((np.arange(n)+start-delay)/HOP,np.arange(len(gains)),gains)
                x*=10**(g/20);wet.seek(start);y=wet.read(n,dtype='float64')
                for a,b in decision['regions']:
                    lo,hi=max(a,start),min(b,start+n)
                    if lo>=hi:continue
                    pos=np.arange(lo,hi);fade=min(1200,(b-a)//2)
                    w=np.minimum(np.clip((pos-a)/max(1,fade),0,1),np.clip((b-1-pos)/max(1,fade),0,1))
                    w=.5-.5*np.cos(np.pi*w);sl=slice(lo-start,hi-start)
                    y[sl]+=w*(x[sl]-y[sl])
                if not np.isfinite(y).all() or np.max(abs(y),initial=0)>=1:raise ValueError('Adaptive output would clip')
                out.write(y)
    timeline=[]
    for a in range(0,len(gains),25):
        b=min(a+25,len(gains));v=decision['voiced_frames'][a:b]
        timeline.append({'start_s':a*.02,'end_s':min(b*.02,features['samples']/48000),
            'source_level_dbfs':float(np.median(features['level_dbfs'][a:b])),
            'pitch_hz':float(np.median(features['pitch_hz'][a:b][v])) if v.any() else None,
            'periodicity':float(np.median(features['periodicity'][a:b])),
            'high_to_mid_db':float(np.median(features['high_to_mid_db'][a:b])),
            'conservative_fraction':float(np.mean(decision['selected'][a:b])),
            'loud_outlier_fraction':float(np.mean(decision['overload_frames'][a:b])),
            'matching_gain_db':float(np.median(gains[a:b]))})
    return {'version':1,'hop_seconds':.02,'analysis_window_seconds':.04,
        'baseline_pitch_hz':decision['baseline_pitch_hz'],
        'regions_seconds':[[a/48000,b/48000] for a,b in decision['regions']],
        'conservative_seconds':sum(b-a for a,b in decision['regions'])/48000,
        'gentle_eq_profile':profile,'alignment_samples':delay,
        'timeline':timeline,
        'analysis_summary':{'pitch_hz_percentiles':np.percentile(features['pitch_hz'][decision['voiced_frames']],[10,50,90]).tolist() if decision['voiced_frames'].any() else [],
                            'speech_threshold_dbfs':decision['speech_threshold_dbfs']},
        'limitations':'Local heuristic, not an AI listening verdict. Register is a routing hint. No transcript, word locations, audio upload or reference recording is used.'}

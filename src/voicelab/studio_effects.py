"""Room prediction, bounded cleanup and constant noise for the studio recipe."""
from collections import deque
from functools import partial
from pathlib import Path
import math,shutil,subprocess
import numpy as np
import soundfile as sf
from scipy import signal,ndimage
from . import dsp,parallel,studio_audio as audio
from .streaming import windows

RATE=48000


def _stop(process):
    if process.poll() is None:
        process.terminate()
        try:process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill();process.wait()


def run_child(command):
    """Stop a local model process before removing its temporary audio on abort."""
    process=subprocess.Popen(command)
    try:
        status=process.wait()
        if status:raise subprocess.CalledProcessError(status,command)
    except BaseException:
        _stop(process)
        raise


def run_children(commands,limit):
    """Run independent local processes, at most ``limit`` at once; stop all on abort."""
    waiting=list(commands);running=[]
    try:
        while waiting or running:
            while waiting and len(running)<limit:
                command=waiting.pop(0);running.append((subprocess.Popen(command),command))
            process,command=running[0]
            status=process.wait()
            running.pop(0)
            if status:raise subprocess.CalledProcessError(status,command)
    except BaseException:
        for process,_ in running:_stop(process)
        raise


class Child:
    """A local model process that runs beside other work."""
    def __init__(self,command,env=None):
        self.command=command;self.process=subprocess.Popen(command,env=env)
    def wait(self):
        status=self.process.wait()
        if status:raise subprocess.CalledProcessError(status,self.command)


class Children:
    """Start background model processes; any still running are stopped on exit."""
    def __init__(self):self.started=[]
    def start(self,command,env=None):
        child=Child(command,env);self.started.append(child);return child
    def __enter__(self):return self
    def __exit__(self,*exc):
        for child in self.started:_stop(child.process)


def room_block(x,high_focus=False):
    from nara_wpe.wpe import wpe_v8
    fft,hop=(1024,256) if high_focus else (2048,512)
    _,_,z=signal.stft(x,RATE,nperseg=fft,noverlap=fft-hop)
    dry=wpe_v8(z[:,None,:],taps=18 if high_focus else 20,delay=2 if high_focus else 3,
               iterations=3,psd_context=1)[:,0,:]
    f=np.fft.rfftfreq(fft,1/RATE)
    weight=np.clip((f-(800 if high_focus else 150))/(1200 if high_focus else 400),0,1)[:,None]*.9
    result=z+weight*(dry-z)
    if high_focus:
        g=np.clip(abs(result)/np.maximum(abs(z),1e-20),.35,1.05)
        result=g*abs(z)*np.exp(1j*np.angle(result))
    else:
        result*=np.minimum(1,1.12*abs(z)/np.maximum(abs(result),1e-20))
    _,y=signal.istft(result,RATE,nperseg=fft,noverlap=fft-hop)
    return y[:len(x)]


def room_file(source,target,high_focus=False,subtype='DOUBLE',progress=lambda *a:None):
    hop=256 if high_focus else 512
    chunk=20 if high_focus else round(20*RATE/hop)*hop/RATE
    context=1 if high_focus else round(2*RATE/hop)*hop/RATE
    return audio.transform(source,target,partial(room_block,high_focus=high_focus),
                           context=context,chunk=chunk,subtype=subtype,workers=True)


DENOISE_PREROLL=20*RATE      # model state settles on real audio before a segment starts
DENOISE_POSTROLL=2*RATE      # look-ahead and crossfade material after it ends
DENOISE_FADE=RATE//20        # 50 ms equal-sum crossfade at each join
DENOISE_MINIMUM=120*RATE     # shortest segment worth a process of its own


def denoise_plan(frames,segments):
    """Segment edges for ``segments`` parallel runs; one run for short takes."""
    count=max(1,min(int(segments),frames//DENOISE_MINIMUM))
    return [round(i*frames/count) for i in range(count+1)]


def _padded_copy(source,target,start,stop):
    with sf.SoundFile(source) as f,audio.writer(target,stop-start+2*RATE,'FLOAT') as out:
        out.write(np.zeros(RATE));f.seek(start);remaining=stop-start
        while remaining:
            x=f.read(min(audio.BLOCK,remaining));remaining-=len(x);out.write(x)
        out.write(np.zeros(RATE))


def denoise(source,target,binary,workdir,segments=1):
    """Run the pinned streaming executable with real context after the take.

    ``segments=1`` is the approved single pass. More segments run the same
    executable on consecutive parts of the take at once: every part starts
    20 s early so the model state has settled, and joins are crossfaded over
    50 ms. That output is close to, but not identical with, the single pass.
    """
    root=Path(workdir);root.mkdir(exist_ok=True,parents=True)
    folder=root/'model-output';frames=sf.info(source).frames
    edges=denoise_plan(frames,segments);count=len(edges)-1
    parts=[]
    for i in range(count):
        a=max(0,edges[i]-DENOISE_PREROLL) if i else 0
        b=min(frames,edges[i+1]+DENOISE_POSTROLL) if i<count-1 else frames
        path=root/('padded.wav' if count==1 else f'padded-{i}.wav')
        _padded_copy(source,path,a,b);parts.append((path,a,b))
    commands=[[str(binary),'-D','-a','9','-o',str(folder),str(path)] for path,_,_ in parts]
    if count==1:run_child(commands[0])
    else:run_children(commands,count)
    outputs=[];native_frames=0;subtype=None
    try:
        for path,a,b in parts:
            f=sf.SoundFile(folder/path.name);outputs.append(f)
            if f.samplerate!=RATE or f.channels!=1 or f.frames<RATE+b-a:
                raise ValueError('Noise model returned insufficient context')
            native_frames+=f.frames;subtype=f.subtype
        def read(i,start,stop):
            outputs[i].seek(RATE+start-parts[i][1]);return outputs[i].read(stop-start)
        fade=DENOISE_FADE
        with audio.writer(target,frames,'FLOAT') as out:
            for i in range(count):
                start=edges[i]+fade if i else 0
                stop=edges[i+1]-fade if i<count-1 else frames
                for position in range(start,stop,audio.BLOCK):
                    out.write(read(i,position,min(stop,position+audio.BLOCK)))
                if i<count-1:
                    ramp=(np.arange(2*fade)+.5)/(2*fade)
                    out.write(read(i,stop,stop+2*fade)*(1-ramp)+read(i+1,stop,stop+2*fade)*ramp)
    finally:
        for f in outputs:f.close()
    shutil.rmtree(root)
    info=dict(model_output_samples=native_frames,retained_samples=[RATE,RATE+frames],
              model_output_subtype=subtype)
    if count>1:
        info.update(retained_samples=None,segments=count,segment_edges=edges,
                    preroll_seconds=DENOISE_PREROLL/RATE,crossfade_seconds=2*fade/RATE)
    return info


CLEANUP_FFT=2048;CLEANUP_HOP=512


def _cleanup_block(x,r,w,offset,first,activity):
    """One aligned cleanup window. ``activity`` holds the frames from ``first``."""
    fft,hop=CLEANUP_FFT,CLEANUP_HOP
    _,t,zr=signal.stft(r,RATE,nperseg=fft,noverlap=fft-hop)
    _,_,zw=signal.stft(w,RATE,nperseg=fft,noverlap=fft-hop)
    _,_,zx=signal.stft(x,RATE,nperseg=fft,noverlap=fft-hop)
    db=20*np.log10(np.maximum(abs(zw),1e-12)/np.maximum(abs(zr),1e-12))
    active=np.interp((offset/RATE+t)/.01,first+np.arange(len(activity)),activity)
    gain=np.clip(ndimage.gaussian_filter(db,(1.5,2)),-(9-7*active)[None,:],0)
    _,y=signal.istft(zx*10**(gain/20),RATE,nperseg=fft,noverlap=fft-hop)
    return y[:len(x)]


def cleanup(source,guide,model,target,speech):
    """The fixed20% guide estimates cleanup; the selected blend keeps its phase."""
    activity=ndimage.gaussian_filter1d(ndimage.maximum_filter1d(speech.astype(float),21),6,mode='nearest')
    hop=CLEANUP_HOP
    chunk=round(30*RATE/hop)*hop/RATE;context=round(.3*RATE/hop)*hop/RATE
    frames=sf.info(source).frames
    if any(sf.info(p).frames!=frames for p in (guide,model)):
        raise ValueError('Cleanup paths have different clocks')
    with audio.writer(target,frames,'FLOAT') as out:
        placed=deque()
        def tasks():
            for new,reference,wet in zip(windows(source,chunk,context),windows(guide,chunk,context),windows(model,chunk,context)):
                start,end,x,crop=new;s,e,r,c=reference;ss,ee,w,cc=wet
                if (start,end)!=(s,e) or (start,end)!=(ss,ee):raise ValueError('Cleanup clock mismatch')
                offset=start-crop.start
                # Each window interpolates activity only between its own frames; hand
                # the worker that span on its original index, not the whole take.
                first=max(0,offset//480-1);last=(offset+len(x))//480+8
                placed.append((start,end,crop,len(x)))
                yield x,r,w,offset,first,activity[first:last]
        for y in parallel.imap(_cleanup_block,tasks()):
            start,end,crop,length=placed.popleft()
            y=y[crop]
            if len(y)!=end-start or not np.isfinite(y).all():raise ValueError('Invalid cleanup output')
            out.write(y)
    return Path(target)


REGION_LOW=signal.butter(3,[300,3000],fs=RATE,btype='bandpass',output='sos')
REGION_HIGH=signal.butter(3,[4500,14000],fs=RATE,btype='bandpass',output='sos')


def _region_features(x,first,last):
    c=slice(first,last)
    return (dsp.envelope(signal.sosfiltfilt(REGION_LOW,x))[c],
            dsp.envelope(signal.sosfiltfilt(REGION_HIGH,x))[c],
            (ndimage.maximum_filter1d((abs(x)>=.98).astype(np.uint8),2401)[::480]>0)[c])


def detect_regions(source,early):
    """Use the existing signal-driven consonant detector; no private word times."""
    from .consonant_plan import plan
    def features(path):
        tasks=((x,crop.start//480,math.ceil(crop.stop/480)) for _,_,x,crop in windows(path,30,4))
        lows,highs,clipped=zip(*parallel.imap(_region_features,tasks))
        return np.concatenate(lows),np.concatenate(highs),np.concatenate(clipped)
    sl,sh,clipped=features(source);wl,wh,_=features(early)
    with sf.SoundFile(source) as f:
        def safe(a,b):
            f.seek(a);return np.max(abs(f.read(b-a)),initial=0)<.98
        regions,_,_=plan(sl,wl,sh,wh,clipped,sf.info(source).frames,True,safe)
    return regions


def repair_weight(early,repair,speech,regions,mapping):
    frames=(sf.info(early).frames+479)//480;weight=np.zeros(frames)
    for m in mapping:
        for lo,hi in regions:
            a=max(lo,m['source_start_sample']);b=min(hi,m['source_end_sample'])
            if a<b:
                a=m['output_start_sample']+a-m['source_start_sample']
                b=m['output_start_sample']+b-m['source_start_sample']
                weight[max(0,a//480):min(frames,(b+479)//480)]=1
    ee=audio.envelopes(early,.08);er=audio.envelopes(repair,.08)
    missing=speech[:len(ee)]&(er>ee*10**(10/20))&(er>10**(-42/20))
    weight=np.maximum(weight,missing.astype(float))
    return ndimage.gaussian_filter1d(ndimage.maximum_filter1d(weight,7),3,mode='nearest')


def crossblend(early,repair,target,weight):
    frames=sf.info(early).frames
    with sf.SoundFile(early) as a,sf.SoundFile(repair) as b,audio.writer(target,frames) as out:
        for start in range(0,frames,audio.BLOCK):
            x=a.read(audio.BLOCK);y=b.read(len(x))
            w=np.interp((start+np.arange(len(x)))/480,np.arange(len(weight)),weight)
            out.write(x*(1-w)+y*w)
    return Path(target)


def noise_bed(target,frames,settings,workdir):
    """Deterministic shaped noise, never reference waveform samples or reverb."""
    taps=np.asarray(settings['taps'],dtype=float)
    if taps.ndim!=1 or len(taps)!=2049 or not np.isfinite(taps).all():
        raise ValueError('Noise filter must contain2049 finite taps')
    rng=np.random.default_rng(settings['seed']);white=Path(workdir)/'white.wav'
    filtered=Path(workdir)/'noise-filtered.wav';total=frames+4096
    with audio.writer(white,total) as out:
        for start in range(0,total,audio.BLOCK):out.write(rng.standard_normal(min(audio.BLOCK,total-start)))
    audio.transform(white,filtered,partial(signal.fftconvolve,in2=taps,mode='same'),context=.1,workers=True)
    energy=0.
    with sf.SoundFile(filtered) as f:
        f.seek(2048)
        for start in range(0,frames,audio.BLOCK):
            x=f.read(min(audio.BLOCK,frames-start));energy+=float(np.sum(x*x))
    factor=10**(settings['base_rms_dbfs']/20)/math.sqrt(energy/frames)
    with sf.SoundFile(filtered) as f,audio.writer(target,frames) as out:
        f.seek(2048)
        for start in range(0,frames,audio.BLOCK):
            x=f.read(min(audio.BLOCK,frames-start))*factor
            positions=start+np.arange(len(x));edge=384
            fade=np.minimum(np.clip(positions/(edge-1),0,1),np.clip((frames-1-positions)/(edge-1),0,1))
            out.write(x*fade)
    white.unlink();filtered.unlink()
    return Path(target)

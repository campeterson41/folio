"""Room prediction, bounded cleanup and constant noise for the studio recipe."""
from pathlib import Path
import math,subprocess
import numpy as np
import soundfile as sf
from scipy import signal,ndimage
from . import dsp,studio_audio as audio
from .streaming import windows

RATE=48000


def run_child(command):
    """Stop a local model process before removing its temporary audio on abort."""
    process=subprocess.Popen(command)
    try:
        status=process.wait()
        if status:raise subprocess.CalledProcessError(status,command)
    except BaseException:
        if process.poll() is None:
            process.terminate()
            try:process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill();process.wait()
        raise


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
    return audio.transform(source,target,lambda x:room_block(x,high_focus),
                           context=context,chunk=chunk,subtype=subtype)


def denoise(source,target,binary,workdir):
    """Run the pinned streaming executable once, with real context after the take."""
    root=Path(workdir);root.mkdir(exist_ok=True,parents=True)
    padded=root/'padded.wav';folder=root/'model-output';frames=sf.info(source).frames
    with sf.SoundFile(source) as f,audio.writer(padded,frames+2*RATE,'FLOAT') as out:
        out.write(np.zeros(RATE))
        for x in f.blocks(audio.BLOCK):out.write(x)
        out.write(np.zeros(RATE))
    run_child([str(binary),'-D','-a','9','-o',str(folder),str(padded)])
    native=folder/padded.name
    with sf.SoundFile(native) as f,audio.writer(target,frames,'FLOAT') as out:
        if f.samplerate!=RATE or f.channels!=1 or f.frames<RATE+frames:
            raise ValueError('Noise model returned insufficient context')
        native_frames=f.frames;f.seek(RATE)
        for start in range(0,frames,audio.BLOCK):out.write(f.read(min(audio.BLOCK,frames-start)))
    return dict(model_output_samples=native_frames,retained_samples=[RATE,RATE+frames],
                model_output_subtype=sf.info(native).subtype)


def cleanup(source,guide,model,target,speech):
    """The fixed20% guide estimates cleanup; the selected blend keeps its phase."""
    activity=ndimage.gaussian_filter1d(ndimage.maximum_filter1d(speech.astype(float),21),6,mode='nearest')
    fft=2048;hop=512
    chunk=round(30*RATE/hop)*hop/RATE;context=round(.3*RATE/hop)*hop/RATE
    frames=sf.info(source).frames
    if any(sf.info(p).frames!=frames for p in (guide,model)):
        raise ValueError('Cleanup paths have different clocks')
    with audio.writer(target,frames,'FLOAT') as out:
        for new,reference,wet in zip(windows(source,chunk,context),windows(guide,chunk,context),windows(model,chunk,context)):
            start,end,x,crop=new;s,e,r,c=reference;ss,ee,w,cc=wet
            if (start,end)!=(s,e) or (start,end)!=(ss,ee):raise ValueError('Cleanup clock mismatch')
            _,t,zr=signal.stft(r,RATE,nperseg=fft,noverlap=fft-hop)
            _,_,zw=signal.stft(w,RATE,nperseg=fft,noverlap=fft-hop)
            _,_,zx=signal.stft(x,RATE,nperseg=fft,noverlap=fft-hop)
            db=20*np.log10(np.maximum(abs(zw),1e-12)/np.maximum(abs(zr),1e-12))
            active=np.interp(((start-crop.start)/RATE+t)/.01,np.arange(len(activity)),activity)
            gain=np.clip(ndimage.gaussian_filter(db,(1.5,2)),-(9-7*active)[None,:],0)
            _,y=signal.istft(zx*10**(gain/20),RATE,nperseg=fft,noverlap=fft-hop)
            y=y[:len(x)][crop]
            if len(y)!=end-start or not np.isfinite(y).all():raise ValueError('Invalid cleanup output')
            out.write(y)
    return Path(target)


def detect_regions(source,early):
    """Use the existing signal-driven consonant detector; no private word times."""
    from .consonant_plan import plan
    low=signal.butter(3,[300,3000],fs=RATE,btype='bandpass',output='sos')
    high=signal.butter(3,[4500,14000],fs=RATE,btype='bandpass',output='sos')
    def features(path):
        lows=[];highs=[];clipped=[]
        for _,_,x,crop in windows(path,30,4):
            c=slice(crop.start//480,math.ceil(crop.stop/480))
            lows.append(dsp.envelope(signal.sosfiltfilt(low,x))[c])
            highs.append(dsp.envelope(signal.sosfiltfilt(high,x))[c])
            clipped.append((ndimage.maximum_filter1d((abs(x)>=.98).astype(np.uint8),2401)[::480]>0)[c])
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
    audio.transform(white,filtered,lambda x:signal.fftconvolve(x,taps,mode='same'),context=.1)
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

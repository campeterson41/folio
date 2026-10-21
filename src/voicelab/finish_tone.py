"""Small optional final bass shelf, with filter state retained across disk blocks."""
from pathlib import Path
import shutil
import numpy as np
import soundfile as sf
from scipy import signal


def bass_sos(gain_db,frequency=160.,rate=48000):
    if not np.isfinite(gain_db) or not -3<=gain_db<=3:
        raise ValueError('Bass adjustment must be between -3 and +3 dB')
    # RBJ low-shelf biquad, shelf slope S=1. The turnover is half the shelf gain.
    a=10**(gain_db/40);w=2*np.pi*frequency/rate;c=np.cos(w)
    z=2*np.sqrt(a)*np.sin(w)/np.sqrt(2)
    b=np.array([a*((a+1)-(a-1)*c+z),2*a*((a-1)-(a+1)*c),a*((a+1)-(a-1)*c-z)])
    d=np.array([(a+1)+(a-1)*c+z,-2*((a-1)+(a+1)*c),(a+1)+(a-1)*c-z])
    return np.array([np.r_[b/d[0],d/d[0]]])


def bass_file(source,target,gain_db=.5,chunk_seconds=30):
    source,target=Path(source),Path(target)
    if source.resolve()==target.resolve():raise ValueError('Cannot overwrite input')
    if target.exists():raise FileExistsError(target)
    if not np.isfinite(chunk_seconds) or chunk_seconds<=0:raise ValueError('Invalid chunk size')
    sos=bass_sos(gain_db)
    info=sf.info(source)
    if info.samplerate!=48000 or info.channels!=1:raise ValueError('Expected 48 kHz mono audio')
    target.parent.mkdir(parents=True,exist_ok=True)
    if gain_db==0:
        with source.open('rb') as inp,target.open('xb') as out:
            shutil.copyfileobj(inp,out,1048576)
        return {'gain_db':0.,'frequency_hz':160.,'samples':info.frames,'bypassed':True}
    state=np.zeros((1,2));count=0;peak=0.
    fmt='RF64' if info.frames*3>3900000000 else 'WAV'
    created=False
    try:
        with sf.SoundFile(source) as inp,sf.SoundFile(target,'x',samplerate=48000,
                channels=1,subtype='PCM_24',format=fmt) as out:
            created=True
            while True:
                x=inp.read(max(1,round(chunk_seconds*48000)),dtype='float64')
                if not len(x):break
                if not np.isfinite(x).all():raise ValueError('Non-finite audio')
                y,state=signal.sosfilt(sos,x,zi=state)
                peak=max(peak,float(np.max(abs(y),initial=0)))
                if peak>=1:raise ValueError('Bass adjustment would clip; provide more headroom')
                out.write(y);count+=len(y)
    except Exception:
        # Only remove a file this invocation created; never an existing output.
        if created:target.unlink(missing_ok=True)
        raise
    if count!=info.frames:raise RuntimeError('Bass adjustment changed duration')
    return {'gain_db':float(gain_db),'frequency_hz':160.,'shelf_slope':1.,
            'samples':count,'sample_peak_dbfs':float(20*np.log10(max(peak,1e-12))),
            'note':'Final low shelf only; no new restoration, leveling or phrase edits.'}

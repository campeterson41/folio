"""Measured, utterance-constant limits on changes to an accepted audio base."""
import math
import numpy as np
import soundfile as sf
from scipy import signal, ndimage
from .adaptive import _read


def fade(positions, start, end, rate=48000):
    n = min(round(.12 * rate), (end - start) // 2)
    w = np.minimum(np.clip((positions-start)/max(1,n),0,1),
                   np.clip((end-1-positions)/max(1,n),0,1))
    return .5-.5*np.cos(np.pi*w)


def choose_mix(base, candidate, regions, gain_db, delay_samples,
               audible_floor_dbfs=-70., maximum_loss_db=1., maximum_boost_db=3.,
               chunk_seconds=30):
    """Choose one weight per region; never switch at a detected bad frame.

    Test the ACTUAL crossfade (including phase cancellation) against the accepted
    base in full-band and300–3000Hz speech energy. A bad frame reduces the weight
    of the entire proposed utterance. Zero means an exact base bypass.
    """
    if not (np.isfinite([gain_db,audible_floor_dbfs,maximum_loss_db,maximum_boost_db]).all()
            and 0<=maximum_loss_db<=3 and 0<=maximum_boost_db<=6 and chunk_seconds>=1):
        raise ValueError('Invalid continuity limits')
    choices=np.array([1.,.75,.5,.25,.1,0.])
    rate=48000;window=960;hop=240;context=4800
    floor=10**(audible_floor_dbfs/10)
    sos=signal.butter(3,[300,3000],fs=rate,btype='bandpass',output='sos')
    results=[]
    with sf.SoundFile(base) as wet,sf.SoundFile(candidate) as dry:
        if (wet.samplerate!=rate or dry.samplerate!=rate or wet.channels!=1 or
                dry.channels!=1 or len(wet)!=len(dry)):
            raise ValueError('Expected matched48kHz mono paths')
        for a,b in regions:
            if not 0<=a<b<=len(wet):raise ValueError('Invalid protected region')
            good=np.ones(len(choices),dtype=bool)
            low=np.full((len(choices),2),np.inf)
            high=np.full((len(choices),2),-np.inf)
            inspected=np.zeros(2,dtype=int)
            for lo in range(a,b,round(chunk_seconds*rate)):
                hi=min(b,lo+round(chunk_seconds*rate))
                start=lo-context;n=hi-lo+2*context
                y=_read(wet,start,n)
                x=_read(dry,start-delay_samples,n)*10**(gain_db/20)
                delta=fade(np.arange(n)+start,a,b)*(x-y)
                pick=np.arange(math.ceil(lo/hop)*hop,hi,hop)-start
                pairs=[(y,delta),(signal.sosfiltfilt(sos,y),signal.sosfiltfilt(sos,delta))]
                for band,(z,d) in enumerate(pairs):
                    avg=lambda v: ndimage.uniform_filter1d(v,window,mode='constant')[pick]
                    energy=avg(z*z);cross=avg(z*d);change=avg(d*d)
                    active=energy>floor
                    if not active.any():continue
                    inspected[band]+=int(active.sum())
                    energy,cross,change=[v[active] for v in (energy,cross,change)]
                    for i,mix in enumerate(choices):
                        ratios=10*np.log10(np.maximum(energy+2*mix*cross+mix*mix*change,1e-30)/energy)
                        low[i,band]=min(low[i,band],float(ratios.min()))
                        high[i,band]=max(high[i,band],float(ratios.max()))
                        good[i]&=bool((ratios>=-maximum_loss_db+1e-6).all() and
                                      (ratios<=maximum_boost_db-1e-6).all())
            # Base bypass is always valid, including recordings with no audible frames.
            good[-1]=True;i=int(np.flatnonzero(good)[0])
            measured=lambda values:[float(v) if np.isfinite(v) else None for v in values]
            results.append({'start_seconds':a/rate,'end_seconds':b/rate,
                'mix':float(choices[i]),'audible_frames_full_and_speech':inspected.tolist(),
                'proposed_min_change_db':measured(low[0]),
                'proposed_max_change_db':measured(high[0]),
                'applied_min_change_db':measured(low[i]),
                'applied_max_change_db':measured(high[i])})
    return {'window_seconds':window/rate,'hop_seconds':hop/rate,
        'audible_floor_dbfs':audible_floor_dbfs,'maximum_loss_db':maximum_loss_db,
        'maximum_boost_db':maximum_boost_db,'bands':['full','300-3000Hz'],
        'regions':results,'note':'One fixed mix per complete region. This bounds introduced level dips/boosts relative to the accepted base, not perceptual quality.'}

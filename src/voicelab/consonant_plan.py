"""Shared global decisions for array and file-backed consonant preservation."""
import numpy as np
from scipy import ndimage
from . import dsp


def plan(sl,wl,sh,wh,clipped,samples,voiced_fricatives,region_safe):
    if voiced_fricatives:
        floor_body=float(np.percentile(sl,15));ceiling=float(np.percentile(sl,95))
        threshold=max(floor_body*5,ceiling*.035,1e-7)
        if ceiling<max(floor_body*4,1e-7):
            threshold=min(threshold,float(np.percentile(sl,85))*.45)
        body=(ndimage.maximum_filter1d((sl>threshold).astype(np.uint8),5)>0)&(sh<sl)
    else:
        body=dsp.speech_mask_energy(sl)&(sh<sl)
    indexes=np.flatnonzero(body)
    if len(indexes)<3:return [],body,indexes
    curve=gain_curve(sl,wl,indexes)
    matched_high=sh*10**(curve/20)
    floor=max(float(np.percentile(sh,10)),1e-9)
    near=ndimage.maximum_filter1d(body.astype(np.uint8),41)>0
    gate=(sh>sl)&(sh>floor*10**(15/20))&(matched_high>wh*10**(4/20))&near
    if voiced_fricatives:
        gate|=(sh>sl*10**(-6/20))&(sh>floor*10**(20/20))&(matched_high>wh*10**(12/20))&near
    gate&=~clipped[:len(gate)]
    gate=ndimage.binary_closing(gate,structure=np.ones(3))
    labels,_=ndimage.label(gate)
    regions=[]
    for obj in ndimage.find_objects(labels):
        if obj is None:continue
        span=obj[0]
        if not 3<=span.stop-span.start<=50:continue
        a=max(0,(span.start-6)*480)
        b=min(samples,(span.stop-1+15)*480)
        if not region_safe(a,b):continue
        if regions and a<=regions[-1][1]:regions[-1][1]=max(b,regions[-1][1])
        else:regions.append([a,b])
    return regions,body,indexes


def gain_curve(source_energy,wet_energy,indexes):
    gains=20*np.log10(np.maximum(wet_energy[indexes],1e-10)/
                     np.maximum(source_energy[indexes],1e-10))
    anchor=float(np.median(gains))
    curve=np.interp(np.arange(len(source_energy)),indexes,np.clip(gains,anchor-6,anchor+6))
    return ndimage.gaussian_filter1d(curve,20)


def source_profile(body,regions,samples,read_source,read_wet):
    usable=body.copy()
    for a,b in regions:usable[a//480:(b+479)//480]=False
    starts=np.flatnonzero(usable)*480
    starts=starts[starts+4096<=samples]
    if len(starts)<3:return None
    starts=starts[np.linspace(0,len(starts)-1,min(len(starts),256)).astype(int)]
    window=np.hanning(4096)
    def spectrum(reader):
        return np.mean([abs(np.fft.rfft(reader(int(a),4096)*window))**2 for a in starts],axis=0)
    freq=np.fft.rfftfreq(4096,1/48000)
    ps,pw=spectrum(read_source),spectrum(read_wet)
    centers=np.geomspace(60,18000,48);diff=[]
    for c in centers:
        bins=(freq>=c/2**(1/6))&(freq<c*2**(1/6))
        if not bins.any():bins[np.argmin(abs(freq-c))]=True
        diff.append(10*np.log10(max(float(pw[bins].mean()),1e-16)/max(float(ps[bins].mean()),1e-16)))
    diff=np.asarray(diff)
    diff-=np.mean(diff[(centers>=150)&(centers<=8000)])
    diff=np.clip(ndimage.gaussian_filter1d(diff,1.5),-9,9)
    diff[centers<75]=min(0.,diff[np.argmin(abs(centers-75))])
    return {'frequencies_hz':centers.tolist(),'gains_db':diff.tolist()}

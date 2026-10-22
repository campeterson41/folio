"""The approved local Folio studio recipe, from raw audio to the 22% blend.

No comparison recording, transcript, manual repair times or private audio is
required. A user-supplied keep list is optional; without one the clock is kept.
"""
from pathlib import Path
import math
import numpy as np
import soundfile as sf
from scipy import signal,ndimage
from . import dsp,source_pipeline,studio_audio as A,studio_effects as E
from .f_master_file import master_file
from .streaming import windows


class Renderer:
    def __init__(self,workdir,preset,progress=None):
        self.root=Path(workdir);self.preset=preset;self.steps=[]
        self.progress=progress or (lambda *a:None)

    def path(self,name):
        return self.root/(name+'.wav')

    def announce(self,label):
        self.progress(label,0)

    def tone(self,source,name,profile,subtype='DOUBLE'):
        return A.transform(source,self.path(name),lambda x:dsp.apply_tone(x,profile),subtype=subtype)

    def norm(self,source,name,speech,target=-23,subtype='DOUBLE'):
        return A.normalize(source,self.path(name),speech,target,subtype)

    def limited(self,source,name,ceiling,release=70,subtype='DOUBLE'):
        A.limit(source,self.path(name),ceiling,release,subtype);return self.path(name)

    def source_voice(self,source,speech,edits,preview_gain=None):
        p=self.preset;self.announce('source voice')
        work=self.root/'source';work.mkdir()
        warm,details=source_pipeline.render(source,work,p['source'],tail_db=0,bass_db=3,progress=self.progress)
        pre=self.path('source-preview-gain')
        if preview_gain is None:
            preview_gain=A.preview_gain(source,warm,p['source_preview']['target_dbfs'])
        A.gain(warm,pre,preview_gain)
        details['preview_gain_db']=preview_gain
        details['preview_gain_method']='saved-override' if self.saved_preview_gain else 'robust-percentile-speech'
        preview=self.limited(pre,'source-preview',p['source_preview']['peak_db'],100)
        peak=20*math.log10(max(A.stats(preview)['true_peak'],1e-12))
        ceiling=p['source_preview']['true_peak_db']
        if peak>ceiling:
            preview=self.limited(preview,'source-preview-second',ceiling-(peak-ceiling)-1,100)
        if A.stats(preview)['true_peak']>10**((ceiling+.001)/20):
            raise ValueError('Source preview exceeds true-peak ceiling')
        quantized=self.path('source-preview-pcm');A.gain(preview,quantized,0,'PCM_24');preview=quantized
        cut=self.path('source-cut');A.cut(preview,cut,edits)
        toned=self.tone(cut,'source-tone',p['ordinary_tone'])
        env=A.envelopes(toned,.16);level=20*np.log10(np.maximum(env,1e-12))
        threshold=float(np.percentile(level[speech[:len(env)]],65))
        curve=ndimage.gaussian_filter1d(-np.clip((level-threshold)*(1-1/1.8),0,4),10,mode='nearest')
        A.apply_curve(toned,self.path('source-steady'),curve)
        steady,_=self.norm(self.path('source-steady'),'source-steady-norm',speech,subtype='FLOAT')
        return steady,details

    def source_finish(self,steady,speech,binary):
        p=self.preset;settings=p['voice'];self.announce('room and voice cleanup')
        room=self.path('source-room');E.room_file(steady,room)
        toned=self.tone(room,'source-room-tone',settings['tone'])
        env=A.envelopes(toned,.16);level=20*np.log10(np.maximum(env,1e-12))
        curve=np.interp(level,settings['level_points_dbfs'],settings['gain_points_db'])
        curve=ndimage.gaussian_filter1d(curve,10)
        current=self.path('source-voice-body');A.apply_curve(toned,current,curve)
        for i in range(2):
            current,_=self.norm(current,f'source-level-{i}',speech)
            current=self.limited(current,f'source-peak-{i}',settings['peak_ceiling_dbfs'])
        pre_df,_=self.norm(current,'source-pre-denoise',speech,subtype='PCM_24')
        wet=self.path('source-denoised');df=E.denoise(pre_df,wet,binary,self.root/'source-model')
        mixed=self.path('source-denoised-bass')
        taps=signal.firwin(2049,200,fs=48000,window=('kaiser',7.5))
        with sf.SoundFile(wet) as a,sf.SoundFile(pre_df) as b,A.writer(mixed,sf.info(wet).frames) as out:
            # FIR context spans each disk boundary, preserving the approved bass.
            for start,end,x,crop in windows(pre_df,30,.1):
                a.seek(start-crop.start);w=a.read(len(x))
                out.write((w+signal.fftconvolve(x-w,taps,mode='same'))[crop])
        post_df,df_gain=self.norm(mixed,'source-post-denoise',speech,subtype='PCM_24')
        toned=self.tone(post_df,'source-post-tone',p['post_cleanup_tone'])
        current,_=self.norm(toned,'source-post-tone-level',speech)
        current=self.limited(current,'source-post-tone-peak',settings['peak_ceiling_dbfs'])
        stem12,_=self.norm(current,'source-complete',speech,subtype='PCM_24')
        toned=self.tone(stem12,'source-eq',p['source_eq'])
        target=20*math.log10(A.stats(stem12,speech)['rms'])
        stem13,_=self.norm(toned,'source-eq-level',speech,target,'PCM_24')
        return stem13,pre_df,post_df,df_gain,df

    def dense_voice(self,stem,pre_df,post_df,df_gain,speech):
        p=self.preset;self.announce('vocal density')
        residual=self.path('source-residual')
        A.add(pre_df,post_df,residual,source_gain=10**(df_gain/20),other_gain=-1)
        residual=self.tone(residual,'source-residual-tone',p['post_cleanup_tone'])
        residual=self.tone(residual,'source-residual-eq',p['source_eq'])
        active=ndimage.maximum_filter1d(speech.astype(float),9,mode='nearest')
        slow=fast=0.;curve=np.empty(len(active));a=math.exp(-.01/.45);b=math.exp(-.01/.035)
        for i,value in enumerate(active):
            slow=max(value,slow*a);fast=max(value,fast*b);curve[i]=max(slow-fast,0)
        curve=np.clip(ndimage.gaussian_filter1d(curve,4,mode='nearest')*.6,0,.4)
        env=A.envelopes(residual,.04)
        cap=np.minimum(1,10**(-62/20)/np.maximum(env*curve[:len(env)],1e-12))
        cap=ndimage.gaussian_filter1d(cap,4)
        curve=curve*cap
        relaxed=self.path('relaxed')
        with sf.SoundFile(stem) as base,sf.SoundFile(residual) as side,A.writer(relaxed,sf.info(stem).frames) as out:
            for start in range(0,sf.info(stem).frames,A.BLOCK):
                x=base.read(A.BLOCK);r=side.read(len(x))
                w=np.interp((start+np.arange(len(x)))/480,np.arange(len(curve)),curve)
                out.write(x+r*w)
        compressed=self.path('dense-compressed');c=p['compression']
        details=A.compress(relaxed,compressed,c['ratio'],c['threshold_dbfs'],c['attack_ms'],c['release_ms'],c['knee_db'],c['parallel_wet'])
        target=20*math.log10(A.stats(stem,speech)['rms'])
        current,_=self.norm(compressed,'dense-level',speech,target)
        current=self.limited(current,'dense-peak',p['voice']['peak_ceiling_dbfs'],100)
        dense,_=self.norm(current,'dense',speech,target,'PCM_24')
        return dense,details

    def early_voice(self,source,native,speech,mapping,edits,dense,regions=None):
        p=self.preset;self.announce('early Folio and speech protection')
        full=self.path('early-full')
        master_file(native,full,p['early_master'],progress=self.progress)
        if regions is None:regions=E.detect_regions(source,full)
        edited=self.path('early-edited');A.cut(full,edited,edits)
        early,_=self.norm(edited,'early-level',speech)
        # Match the original intermediate quantization, independently of the
        # double-precision early path used for the actual repair blend.
        A.gain(early,self.path('early-reference'),0,'PCM_24')
        focused=self.path('repair-focused');E.room_file(dense,focused,high_focus=True)
        repair=self.tone(focused,'repair-tone',p['repair_tone'])
        target=20*math.log10(A.stats(early,speech)['rms'])
        repair,_=self.norm(repair,'repair-level',speech,target)
        weight=E.repair_weight(early,repair,speech,regions,mapping)
        blend=self.path('early-repaired');E.crossblend(early,repair,blend,weight)
        blend,_=self.norm(blend,'early-repaired-level',speech)
        foundation=self.limited(blend,'early-foundation',-1.5,100,'FLOAT')
        return foundation,dict(repair_regions=len(regions),repair_fraction=float(np.mean(weight>.5)))

    def early_finish(self,foundation,speech):
        p=self.preset;self.announce('early tone and continuous floor')
        toned=self.tone(foundation,'early-final-tone',p['early_eq'])
        activity=ndimage.maximum_filter1d(speech.astype(float),9)
        held=np.empty(len(activity));state=0.;decay=math.exp(-.01/.3)
        for i,value in enumerate(activity):
            state=max(value,state*decay);held[i]=state
        curve=-6*(1-ndimage.gaussian_filter1d(held,4,mode='nearest'))
        reduced=self.path('early-gap-level');A.apply_curve(toned,reduced,curve)
        voice,_=self.norm(reduced,'early-final-level',speech)
        bed=self.path('noise-bed');E.noise_bed(bed,sf.info(voice).frames,p['noise'],self.root)
        # Keep the original FLOAT noise checkpoint as well as the full precision
        # generated bed; it gives reproducible blending across recipe stages.
        saved=self.path('noise-bed-float');A.gain(bed,saved,0,'FLOAT')
        noisy=self.path('early-with-bed');A.add(voice,bed,noisy)
        early,db=self.norm(noisy,'early',speech,subtype='PCM_24')
        bed_scaled=self.path('noise-bed-scaled');A.gain(saved,bed_scaled,db)
        return early,bed_scaled

    def final_blend(self,dense,early,bed,speech,binary,blend,fuzz):
        self.announce('blending and final protected cleanup')
        dry=self.path('early-without-bed');A.add(early,bed,dry,other_gain=-1)
        guide=self.path('cleanup-guide');A.add(dense,dry,guide,.2,.8,'FLOAT')
        selected=self.path('selected-blend');A.add(dense,dry,selected,blend,1-blend,'FLOAT')
        model=self.path('cleanup-model');info=E.denoise(guide,model,binary,self.root/'final-model')
        cleaned=self.path('cleaned');E.cleanup(selected,guide,model,cleaned,speech)
        target=20*math.log10(A.stats(dense,speech)['rms'])
        voice,db=self.norm(cleaned,'final-voice',speech,target)
        final=self.path('studio-final');A.add(voice,bed,final,other_gain=fuzz,subtype='PCM_24')
        return final,dict(final_voice_gain_db=db,cleanup=info)

    def render(self,source,native,binary,edits=None,blend=None,fuzz=None,regions=None,preview_gain=None):
        p=self.preset
        self.saved_preview_gain=preview_gain is not None
        blend=p['blend_fraction'] if blend is None else blend
        fuzz=p['fuzz_fraction'] if fuzz is None else fuzz
        if not math.isfinite(blend) or not 0<=blend<=1 or not math.isfinite(fuzz) or not 0<=fuzz<=1:
            raise ValueError('Blend and fuzz fractions must be finite and between 0 and 1')
        raw=self.path('original-edited');mapping=A.cut(source,raw,edits)
        speech,mask_details=A.activity_mask(raw)
        if speech.sum()<10:raise ValueError('Not enough speech')
        steady,source_info=self.source_voice(source,speech,edits,preview_gain)
        stem,pre,post,df_gain,denoise_info=self.source_finish(steady,speech,binary)
        dense,density_info=self.dense_voice(stem,pre,post,df_gain,speech)
        foundation,repair_info=self.early_voice(source,native,speech,mapping,edits,dense,regions)
        early,bed=self.early_finish(foundation,speech)
        final,final_info=self.final_blend(dense,early,bed,speech,binary,blend,fuzz)
        metrics=A.stats(final,speech)
        if metrics['samples']!=sf.info(raw).frames or metrics['true_peak']>=.99:
            raise ValueError('Studio output failed clock/peak verification')
        return final,dict(samples=metrics['samples'],metrics=metrics,source=source_info,
            blend_fraction=blend,fuzz_fraction=fuzz,density=density_info,
            speech_protection=repair_info,first_denoise=denoise_info,**final_info,
            speech_edited=edits is not None,reference_audio_read=False,
            **mask_details,speech_fraction=float(np.mean(speech)),
            synthetic_noise_added=fuzz>0,neural_models_used=True)

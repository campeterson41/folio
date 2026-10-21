#!/usr/bin/env python3
"""Local TF-Restormer inference with bounded PCM buffers and a fixed global grid."""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import time

os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('HF_HUB_DISABLE_IMPLICIT_TOKEN','1')
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY','1')
import numpy as np
import soundfile as sf
import torch
from voicelab.chunks import restore_file
from voicelab.streaming import sha
from voicelab.f_artifacts import runtime_root


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    args=ap.parse_args()
    receipt=args.output.with_suffix('.json')
    if args.output.exists() or receipt.exists():raise FileExistsError(args.output)
    info=sf.info(args.input)
    if info.samplerate!=48000 or info.channels!=1 or info.frames<4800:
        raise ValueError('Expected at least 100 ms of 48 kHz mono input')
    runtime=runtime_root()
    sys.path.insert(0,str(runtime/'vendor/TF_Restormer'))
    from tf_restormer import SEInference
    torch.set_num_threads(4)
    torch.manual_seed(37)
    if args.device=='cuda':
        torch.cuda.set_per_process_memory_fraction(.82)
        torch.cuda.reset_peak_memory_stats()
    model=SEInference.from_pretrained(checkpoint_path=str(runtime/'models/tf-restormer'),device=args.device)
    source_hash=sha(args.input)
    start=time.monotonic()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    last=[-30.]
    input_peak=[0.]
    def progress(seconds):
        if seconds-last[0]>=30:
            print(f'Restored {seconds:.1f} / {info.duration:.1f} s',flush=True)
            last[0]=seconds
    def infer(block):
        input_peak[0]=max(input_peak[0],float(np.max(abs(block),initial=0)))
        result=model.process_waveform(torch.tensor(block,dtype=torch.float32)[None,:],
            fs_in=48000,fs_out=48000,mode='single_pass')
        return result['waveform'].detach().cpu().numpy().reshape(-1)
    with tempfile.TemporaryDirectory(prefix='voice-lab-tf-',dir=args.output.parent) as tmp:
        unscaled=Path(tmp)/'native-double.wav'
        with torch.inference_mode():
            stats=restore_file(args.input,unscaled,infer,progress=progress)
        if stats['peak']<1e-9 and input_peak[0]>1e-5:
            raise RuntimeError('Model returned silence instead of input speech')
        gain=.98/stats['peak'] if stats['peak']>=.98 else 1.
        part=Path(tmp)/'export.wav'
        with sf.SoundFile(unscaled) as inp,sf.SoundFile(part,'w',samplerate=48000,
                channels=1,subtype='PCM_24',format='RF64' if info.frames*3>3900000000 else 'WAV') as out:
            while len(x:=inp.read(48000*10,dtype='float64')):
                out.write(x*gain)
        if sha(args.input)!=source_hash:raise RuntimeError('Input changed during inference')
        if sf.info(part).frames!=info.frames:raise RuntimeError('Output duration differs')
        if receipt.exists():raise FileExistsError(receipt)
        os.link(part,args.output)
    result={'id':'tf-file-native','family':'tf-restormer',
        'parameters':{'seconds':3.,'input_rate':48000,'context_seconds':.5,
                      'crossfade_seconds':.08,'device':args.device,'input':str(args.input.resolve())},
        'source_sha256':source_hash,'output_sha256':sha(args.output),
        'samples':info.frames,'runtime_seconds':time.monotonic()-start,
        'write_safety_gain_db':float(20*np.log10(gain)),
        'maximum_input_window_seconds':stats['maximum_input_window_seconds'],
        'gpu_peak_mib':torch.cuda.max_memory_allocated()/2**20 if args.device=='cuda' else 0.,
        'post_processing':'One whole-file peak safety gain only; file-backed global 3s grid.'}
    with receipt.open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    print(receipt,flush=True)


if __name__=='__main__':main()

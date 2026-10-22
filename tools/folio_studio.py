#!/usr/bin/env python3
"""Official Folio Studio: local narration restoration, 22% blend and a quiet floor."""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import soundfile as sf
from voicelab import __version__
from voicelab.studio_pipeline import Renderer
from voicelab.studio_effects import run_child
from voicelab import studio_audio as audio
from voicelab.streaming import sha
from voicelab.conservative import binary
from voicelab.f_artifacts import runtime_root, verify as verify_artifacts
from folio_source import inspect_input, _interrupt
from enhance_f import validate_native


def repair_regions(value, frames):
    if not isinstance(value, dict) or set(value) != {'sample_rate', 'regions'} or value['sample_rate'] != 48000:
        raise ValueError('Repair plan needs sample_rate: 48000 and regions: [[start, end], ...]')
    regions=value['regions']
    if not isinstance(regions,list) or len(regions)>100000:
        raise ValueError('Invalid repair regions')
    last=0
    for pair in regions:
        if (not isinstance(pair,list) or len(pair)!=2 or any(type(x) is not int for x in pair)
                or not 0<=pair[0]<pair[1]<=frames or pair[0]<last):
            raise ValueError('Repair regions must be ordered, nonoverlapping source sample intervals')
        last=pair[1]
    return regions


def preflight_source(source):
    # Reject insufficient room tone before spending time on neural inference.
    import numpy as np
    from voicelab.adaptive import analyze
    from voicelab.source_pipeline import noise_profile
    features=analyze(source)
    _,indexes=noise_profile(source,features)
    if np.percentile(features['level_dbfs'],90)-np.median(features['level_dbfs'][indexes])<15:
        raise ValueError('Quiet regions are within 15 dB of speech; record clean room tone between phrases')


def process(args):
    source=args.input.resolve();output=args.output;receipt=output.with_suffix('.json')
    if output.suffix.lower()!='.wav':raise ValueError('Use a .wav output')
    for p in (output,receipt):
        if p.exists() or p.is_symlink():raise FileExistsError(f'Refusing to overwrite {p}')
    info,input_check=inspect_input(source);source_hash=sha(source)
    preset_hash=sha(args.preset)
    preset=json.loads(args.preset.read_text())
    if preset.get('schema')!='folio-studio/1':raise ValueError('Unsupported studio preset schema')
    edits=json.loads(args.edit_list.read_text()) if args.edit_list else None
    audio.validate_edits(edits,info.frames)
    regions=repair_regions(json.loads(args.repair_regions.read_text()),info.frames) if args.repair_regions else None
    # Conservative preflight: bounded RAM, but many file-backed stages.
    required=info.frames*8*70
    if shutil.disk_usage(output.parent).free<required:
        raise OSError(f'Studio needs about {required/1024**3:.1f} GiB temporary free space for this take')
    runtime=runtime_root();executable=binary(runtime)
    def progress(phase, seconds):
        if not args.quiet:print(f'{phase}: {seconds:.1f}s',file=sys.stderr,flush=True)
    preflight_source(source)
    artifacts=None
    with tempfile.TemporaryDirectory(prefix='.folio-studio-',dir=output.parent) as tmp:
        root=Path(tmp);native=args.native_stem or root/'native.wav'
        if not args.native_stem:
            artifacts=verify_artifacts(runtime)
            run_child([sys.executable,str(ROOT/'tools/restore_tf_file.py'),
                '--input',str(source),'--output',str(native),'--device',args.device])
        native_receipt=validate_native(native,source_hash,info.frames)
        native_hash=sha(native)
        native_receipt=dict(native_receipt)
        native_receipt.pop('path',None)
        native_receipt['parameters']=dict(native_receipt['parameters'])
        native_receipt['parameters'].pop('input',None)
        final,details=Renderer(root,preset,progress).render(source,native,executable,
            edits=edits,regions=regions,preview_gain=args.preview_gain_db,
            blend=args.blend,fuzz=args.fuzz)
        check=sf.info(final);metrics=audio.stats(final)
        if check.samplerate!=48000 or check.channels!=1 or check.subtype!='PCM_24' or metrics['true_peak']>=.99:
            raise ValueError('Output format/peak verification failed')
        if sha(source)!=source_hash or sha(native)!=native_hash or sha(args.preset)!=preset_hash:
            raise RuntimeError('Input changed during processing; nothing published')
        implementation=[Path(__file__).resolve(),ROOT/'tools/restore_tf_file.py',*sorted((ROOT/'src/voicelab').glob('*.py'))]
        metadata=dict(recipe='folio-studio-v1',folio_version=__version__,
            source_sha256=source_hash,output_sha256=sha(final),preset_sha256=preset_hash,
            input_check=input_check,sample_rate=48000,output_subtype='PCM_24',
            native_restoration=native_receipt,fresh_runtime_verification=artifacts,
            denoiser_sha256=sha(executable),
            implementation_sha256={str(p.relative_to(ROOT)):sha(p) for p in implementation},
            repair_mode='supplied-local-plan' if regions is not None else 'automatic-current-detector',
            edit_plan_sha256=sha(args.edit_list) if args.edit_list else None,
            repair_plan_sha256=sha(args.repair_regions) if args.repair_regions else None,
            **details)
        meta=root/'receipt.json';meta.write_text(json.dumps(metadata,indent=2)+'\n')
        try:
            os.link(final,output);os.link(meta,receipt)
        except BaseException:
            for dest,original in ((output,final),(receipt,meta)):
                try:
                    if os.path.samefile(dest,original):dest.unlink()
                except FileNotFoundError:pass
            raise
    return metadata


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input',type=Path,required=True,help='48 kHz mono WAV, at least 5 seconds with quiet pauses')
    ap.add_argument('--output',type=Path,required=True,help='New 24-bit WAV; JSON receipt saved alongside')
    ap.add_argument('--preset',type=Path,default=ROOT/'presets/studio-v1.json')
    ap.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    ap.add_argument('--native-stem',type=Path,help='Reuse a source-hash-verified TF native WAV and its JSON receipt')
    ap.add_argument('--edit-list',type=Path,help='Optional local sample-index keep list; no automatic repeat deletion')
    ap.add_argument('--repair-regions',type=Path,help='Optional local source-sample repair plan for a saved session')
    ap.add_argument('--preview-gain-db',type=float,help='Saved session gain override; normally derived from speech')
    ap.add_argument('--blend',type=float,help='Restoration blend fraction, 0–1 (default .22)')
    ap.add_argument('--fuzz',type=float,help='Synthetic floor fraction, 0–1 (default .31473135; 0 disables)')
    ap.add_argument('--quiet',action='store_true')
    args=ap.parse_args(argv)
    for label,value,low,high in [('blend',args.blend,0,1),('fuzz',args.fuzz,0,1),('preview-gain-db',args.preview_gain_db,-36,36)]:
        if value is not None and (not math.isfinite(value) or not low<=value<=high):
            ap.error(f'--{label} must be finite and between {low} and {high}')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    previous={s:signal.signal(s,_interrupt) for s in (signal.SIGTERM,signal.SIGHUP)}
    lock_path=args.output.parent/('.'+args.output.name+'.lock')
    try:
        with lock_path.open('a') as lock:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:raise FileExistsError('This output is already being processed')
            try:process(args)
            finally:lock_path.unlink(missing_ok=True)
    finally:
        for s,handler in previous.items():signal.signal(s,handler)
    if not args.quiet:print(args.output)


if __name__=='__main__':
    try:main()
    except KeyboardInterrupt:
        print('Interrupted; temporary files removed and nothing published.',file=sys.stderr);sys.exit(130)
    except (ValueError,OSError,RuntimeError,KeyError,subprocess.CalledProcessError) as exc:
        print(f'folio: {exc}',file=sys.stderr);sys.exit(1)

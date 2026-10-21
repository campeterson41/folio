#!/usr/bin/env python3
"""Folio restoration with automatic source-consonant preservation.

Run with the round2 Python environment and PYTHONPATH=src. No Auphonic audio,
timestamps, transcription, API, or manual phrase selection is used. A cached
native stem can be reused only after verifying its source hash and recipe.
"""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import soundfile as sf
from voicelab.f_master_file import master_file
from voicelab.consonant_file import preserve_file
from voicelab.streaming import sha
from voicelab.f_artifacts import verify as verify_artifacts, runtime_root

ROOT = Path(__file__).resolve().parents[1]


def validate_native(stem, source_hash, frames):
    receipt = json.loads(stem.with_suffix('.json').read_text())
    params = receipt['parameters']
    info = sf.info(stem)
    if (receipt['source_sha256'] != source_hash or
        receipt['output_sha256'] != sha(stem) or
        receipt['family'] != 'tf-restormer' or
        params['seconds'] != 3 or params['input_rate'] != 48000 or
        info.frames != frames or info.samplerate != 48000 or info.channels != 1):
        raise ValueError('Cached native stem does not match source and frozen recipe')
    return receipt


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--native-stem', type=Path)
    ap.add_argument('--device', choices=['cuda','cpu'], default='cuda')
    ap.add_argument('--preserve-voiced-fricatives', action='store_true',
                    help='Experimental stronger-loss detector for voiced Z-like sounds')
    ap.add_argument('--finish-levels',action='store_true',
                    help='Optional final RMS/peak control, leaving tone/restoration settings fixed')
    ap.add_argument('--bass-db',type=float,default=0.,
                    help='Optional final 160 Hz bass shelf in dB; try 0.5 for a subtle lift')
    ap.add_argument('--preset', type=Path, default=ROOT/'presets/f-voice-tone-20261021.json')
    args = ap.parse_args()
    if not math.isfinite(args.bass_db) or not -3 <= args.bass_db <= 3:
        ap.error('--bass-db must be between -3 and +3 dB')
    if args.output.suffix.lower() != '.wav':
        raise ValueError('Use a .wav output path')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with (args.output.parent/('.'+args.output.with_suffix('.json').name+'.lock')).open('a') as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            raise FileExistsError('This output is already being processed')
        process(args)


def process(args):
    if args.output.exists() or args.output.with_suffix('.json').exists():
        raise FileExistsError(args.output)
    info = sf.info(args.input)
    if info.samplerate != 48000 or info.channels != 1:
        raise ValueError('This experimental recipe requires 48 kHz mono input')
    source_hash = sha(args.input)
    preset = json.loads(args.preset.read_text())
    artifacts = None
    with tempfile.TemporaryDirectory(prefix='voice-lab-f-',dir=args.output.parent) as tmp:
        stem = args.native_stem or Path(tmp)/'native.wav'
        if args.native_stem is None:
            artifacts = verify_artifacts(runtime_root())
            subprocess.run([sys.executable,str(ROOT/'tools/restore_tf_file.py'),
                            '--device',args.device,
                            '--input',str(args.input.resolve()),
                            '--output',str(stem)],check=True)
        native_receipt = validate_native(stem, source_hash, info.frames)
        baseline = Path(tmp)/'master.wav'
        mastering = master_file(stem,baseline,preset)
        final = Path(tmp)/'final.wav'
        guard = preserve_file(args.input,baseline,final,
                              voiced_fricatives=args.preserve_voiced_fricatives)
        bass = None
        if getattr(args,'bass_db',0):
            from voicelab.finish_tone import bass_file
            bass_target = Path(tmp)/'bass.wav'
            bass = bass_file(final,bass_target,args.bass_db)
            final = bass_target
        finishing = None
        if getattr(args,'finish_levels',False):
            from voicelab.finish_levels import finish_file
            finished = Path(tmp)/'finished.wav'
            settings = preset['mastering_settings']
            finishing = finish_file(final,finished,target_db=settings.get('target_db',-20),
                                     peak_db=settings.get('peak_db',-4))
            final = finished
        if sf.info(final).frames != info.frames:
            raise ValueError('Invalid final result')
        if sha(args.input) != source_hash:
            raise RuntimeError('Source changed during processing')
        metadata = {
            'recipe':('f-auto-consonants-v2' if args.preserve_voiced_fricatives else 'f-auto-consonants-v1') + ('-bass' if bass else '') + ('-levels' if finishing else ''), 'source_sha256':source_hash,
            'output_sha256':sha(final), 'samples':info.frames,
            'preset_sha256':sha(args.preset), 'native_restoration':native_receipt,
            'mastering':mastering, 'automatic_consonant_preservation':guard,
            'level_finishing':finishing, 'bass_adjustment':bass, 'fresh_runtime_verification':artifacts,
            'implementation_sha256':{str(p.relative_to(ROOT)):sha(p) for p in [
                Path(__file__).resolve(),ROOT/'src/voicelab/f_master_file.py',
                ROOT/'src/voicelab/consonant_file.py',ROOT/'src/voicelab/consonant_plan.py',
                ROOT/'src/voicelab/finish_levels.py',ROOT/'src/voicelab/finish_tone.py']},
            'status':'Experimental general processor; human evaluation pending. '
                     'Audio buffers bounded; global 10ms decision arrays scale with duration.',
        }
        meta=Path(tmp)/'receipt.json';meta.write_text(json.dumps(metadata,indent=2)+'\n')
        if args.output.with_suffix('.json').exists():raise FileExistsError('Receipt already exists')
        # Exclusive hard links publish complete files without replacing a
        # recording that appeared while processing. Temp files share this FS.
        os.link(final,args.output)
        os.link(meta,args.output.with_suffix('.json'))
    print(args.output)


if __name__ == '__main__':
    main()

"""Pinned local DeepFilterNet backend for adaptive performance protection."""
import json
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import urllib.request

LOCK=Path(__file__).resolve().parents[2]/'resources/conservative-lock.json'

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def binary(runtime):
    spec=json.loads(LOCK.read_text());path=Path(runtime)/spec['path']
    if not path.is_file():raise FileNotFoundError('Run setup_f_runtime.py --with-conservative for --adaptive')
    if sha(path)!=spec['sha256']:raise ValueError('Conservative binary checksum differs')
    return path

def install(runtime,offline=False):
    spec=json.loads(LOCK.read_text());target=Path(runtime)/spec['path']
    if target.exists():return binary(runtime)
    target.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent,delete=False) as out:
        temporary=Path(out.name)
        try:
            cache=Path.home()/'.local/share/voice-enhancement-lab'/spec['path']
            if cache.is_file() and sha(cache)==spec['sha256']:
                with cache.open('rb') as f:shutil.copyfileobj(f,out,1048576)
            else:
                if offline:raise FileNotFoundError('Pinned conservative binary unavailable offline')
                with urllib.request.urlopen(spec['url'],timeout=120) as f:shutil.copyfileobj(f,out,1048576)
            out.flush()
            if sha(temporary)!=spec['sha256']:raise ValueError('Conservative download checksum differs')
            temporary.chmod(0o755);os.link(temporary,target)
        finally:temporary.unlink(missing_ok=True)
    return binary(runtime)

def validate(stem,source_hash,frames):
    import soundfile as sf
    stem=Path(stem);meta=json.loads(stem.with_suffix('.json').read_text());info=sf.info(stem)
    if (meta.get('recipe')!='deepfilternet3-conservative-native' or
        meta.get('source_sha256')!=source_hash or meta.get('output_sha256')!=sha(stem) or
        meta.get('attenuation_limit_db')!=12 or not meta.get('delay_compensated') or
        info.frames!=frames or info.samplerate!=48000 or info.channels!=1):
        raise ValueError('Conservative stem does not match input or recipe')
    return meta

def restore(source,target,runtime,progress=lambda *a:None):
    import numpy as np
    import soundfile as sf
    from .chunks import restore_file
    executable=binary(runtime);source,target=map(Path,(source,target))
    source_hash=sha(source)
    with tempfile.TemporaryDirectory(prefix='folio-denoise-',dir=target.parent) as tmp:
        root=Path(tmp);inp=root/'input.wav';out=root/'out'
        def infer(x):
            sf.write(inp,np.pad(x,(24000,24000)),48000,subtype='FLOAT')
            subprocess.run([str(executable),'-D','-a','12','-o',str(out),str(inp)],check=True,
                stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            y,rate=sf.read(out/'input.wav')
            if rate!=48000 or len(y)<24000+len(x):raise ValueError('Short conservative output')
            return y[24000:24000+len(x)]
        info=restore_file(source,target,infer,core_seconds=30,context_seconds=1,
                          crossfade_seconds=.08,progress=progress)
    if sha(source)!=source_hash:raise RuntimeError('Input changed during conservative restoration')
    meta={'recipe':'deepfilternet3-conservative-native','source_sha256':source_hash,
          'output_sha256':sha(target),'samples':info['samples'],'attenuation_limit_db':12,
          'delay_compensated':True,'padding_seconds_each_side':.5,
          'maximum_input_window_seconds':info['maximum_input_window_seconds']}
    target.with_suffix('.json').write_text(json.dumps(meta,indent=2)+'\n')
    return meta

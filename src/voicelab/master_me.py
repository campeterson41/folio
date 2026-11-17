"""Pinned upstream master_me engine, hosted offline by FFmpeg's LADSPA filter."""
import functools
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import tempfile
from .f_artifacts import runtime_root

ROOT=Path(__file__).resolve().parents[2]


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


@functools.lru_cache(maxsize=1)
def verify_host():
    executable=shutil.which('ffmpeg')
    if executable is None:
        raise ValueError('master_me requires FFmpeg with LADSPA support')
    filters=subprocess.check_output([executable,'-hide_banner','-filters'],stderr=subprocess.DEVNULL,text=True)
    if not any(line.split()[1:2]==['ladspa'] for line in filters.splitlines()):
        raise ValueError('This FFmpeg build has no LADSPA filter for master_me')
    return executable


def engine(runtime=None):
    verify_host()
    lock=json.loads((ROOT/'resources/mastering-lock.json').read_text())
    directory=(runtime_root() if runtime is None else Path(runtime))/'mastering'
    try:
        manifest=json.loads((directory/'manifest.json').read_text())
        plugin=directory/lock['plugin']
        if any(manifest.get(k)!=v for k,v in lock.items()) or sha(plugin)!=manifest['sha256']:
            raise ValueError('master_me runtime identity mismatch; nothing processed')
    except (OSError,KeyError) as exc:
        raise ValueError('Install the mastering engine: python3 tools/setup_mastering_runtime.py') from exc
    return plugin,manifest


def settings():
    path=ROOT/'presets/mastering-speech-v1.json'
    p=json.loads(path.read_text());symbols=p['control_symbols'];controls=p['controls']
    if (len(symbols)!=61 or len(set(symbols))!=61 or set(symbols)!=set(controls) or
            any(not isinstance(controls[k],(int,float)) or not math.isfinite(controls[k]) for k in symbols)):
        raise ValueError('Invalid master_me speech settings')
    return p,sha(path)


def process(source,target):
    plugin,manifest=engine();p,preset_sha=settings()
    controls='|'.join(str(p['controls'][symbol]) for symbol in p['control_symbols'])
    target=Path(target).resolve()
    # Use a fixed relative library name, keeping paths out of filter syntax.
    with tempfile.TemporaryDirectory(prefix='.master-me-',dir=target.parent) as tmp:
        directory=Path(tmp);library=directory/'engine.so';shutil.copyfile(plugin,library)
        if sha(library)!=manifest['sha256']:
            raise ValueError('Mastering engine changed during processing')
        output=directory/'processed.wav'
        filters=('pan=stereo|c0=c0|c1=c0,'
            'ladspa=file=./engine.so:plugin=master_me:controls='+controls+
            ',pan=mono|c0=0.5*c0+0.5*c1')
        command=[shutil.which('ffmpeg') or 'ffmpeg','-nostdin','-hide_banner','-nostats',
            '-i',str(Path(source).resolve()),'-af',filters,'-ar','48000',
            '-c:a','pcm_f64le','-rf64','auto','-n',str(output)]
        with tempfile.TemporaryFile(mode='w+') as log:
            child=subprocess.Popen(command,cwd=directory,stdout=subprocess.DEVNULL,stderr=log)
            try:
                if child.wait():
                    raise ValueError('master_me rendering failed; no result published')
            except BaseException:
                if child.poll() is None:
                    child.terminate()
                    try:child.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        child.kill();child.wait()
                raise
        # Same-filesystem publication; never replace another writer's result.
        import os
        os.link(output,target)
    return dict(engine='master_me',version=manifest['version'],
        upstream_commit=manifest['commit'],plugin_sha256=manifest['sha256'],
        preset_sha256=preset_sha,base_preset=p['base_preset'],controls=p['controls'],
        host='FFmpeg LADSPA',channel_routing='dual mono into stereo engine; average identical outputs')

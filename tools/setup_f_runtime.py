#!/usr/bin/env python3
"""Install or read-only verify the isolated, pinned F inference runtime.

Public code/weights only. No audio uploads, accounts or credential access.
Existing unrecognized environments/artifacts are preserved rather than replaced.
"""
import argparse
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT/'src'))
from voicelab.f_artifacts import verify

OWNER='voice-lab-f/setup-v1'


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def pins(path):
    result={}
    for line in Path(path).read_text().splitlines():
        line=line.strip()
        if not line or line.startswith('#'):continue
        m=re.fullmatch(r'([A-Za-z0-9_.-]+)==([A-Za-z0-9+_.-]+)',line)
        if not m:raise ValueError('Every inference dependency must have one exact version')
        if m[1] in result:raise ValueError('Duplicate dependency pin')
        result[m[1]]=m[2]
    return result


def run(*args,**kwargs):
    subprocess.run([str(a) for a in args],check=True,**kwargs)


def ensure_uv(runtime,offline=False):
    lock=json.loads((ROOT/'resources/bootstrap-lock.json').read_text())
    target=runtime/'bin/uv'
    if target.exists():
        if sha(target)!=lock['binary_sha256']:raise ValueError('Existing uv binary differs; preserved')
        return target
    target.parent.mkdir(parents=True,exist_ok=True)
    cached=Path.home()/'.local/share/folio/bin/uv'
    if cached.is_file() and sha(cached)==lock['binary_sha256']:
        data=cached.read_bytes()
    else:
        if offline:raise FileNotFoundError('Pinned uv bootstrap is not available locally')
        with urllib.request.urlopen(lock['url'],timeout=90) as response:
            archive=response.read()
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            entry=next(m for m in tar.getmembers() if m.isfile() and m.name.endswith('/uv'))
            data=tar.extractfile(entry).read()
    if hashlib.sha256(data).hexdigest()!=lock['binary_sha256']:
        raise ValueError('uv bootstrap checksum differs')
    with target.open('xb') as f:f.write(data)
    target.chmod(0o755)
    return target


def acquire_artifact(runtime,name,spec,lock,offline=False):
    target=runtime/spec['path']
    if target.exists():
        if sha(target)!=spec['sha256']:raise ValueError(f'Existing {name} differs; preserved')
        return
    target.parent.mkdir(parents=True,exist_ok=True)
    cached=Path.home()/'.local/share/folio'/spec['path']
    with tempfile.NamedTemporaryFile(dir=target.parent,prefix='.f-download-',delete=False) as f:
        temporary=Path(f.name)
        try:
            if cached.is_file() and sha(cached)==spec['sha256']:
                with cached.open('rb') as src:shutil.copyfileobj(src,f,1048576)
            else:
                if offline:raise FileNotFoundError(f'Pinned model artifact unavailable locally: {name}')
                url=f"https://huggingface.co/{lock['model_repo']}/resolve/{lock['model_revision']}/{name}"
                with urllib.request.urlopen(url,timeout=120) as src:shutil.copyfileobj(src,f,1048576)
            f.flush()
            if sha(temporary)!=spec['sha256']:raise ValueError(f'Download checksum differs: {name}')
            os.link(temporary,target)
        finally:
            temporary.unlink(missing_ok=True)


def ensure_source(runtime,lock,offline=False):
    spec=lock['source'];target=runtime/spec['directory']
    if target.exists():
        if not (target/'.git').is_dir():raise ValueError('Unrecognized existing model source preserved')
        revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=target,text=True).strip()
        if revision!=spec['revision']:raise ValueError('Existing model source revision differs; preserved')
        run('git','diff','--exit-code','--quiet','HEAD',cwd=target)
        return
    target.parent.mkdir(parents=True,exist_ok=True)
    cached=Path.home()/'.local/share/folio'/spec['directory']
    origin=spec['origin']
    if (cached/'.git').is_dir():
        revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=cached,text=True).strip()
        if revision==spec['revision']:origin=str(cached)
    if offline and origin==spec['origin']:
        raise FileNotFoundError('Pinned model source unavailable locally')
    with tempfile.TemporaryDirectory(dir=target.parent,prefix='.f-source-') as tmp:
        checkout=Path(tmp)/'source'
        run('git','clone','--no-checkout',origin,checkout)
        run('git','checkout','--detach',spec['revision'],cwd=checkout)
        run('git','remote','set-url','origin',spec['origin'],cwd=checkout)
        checkout.rename(target)


def installed_versions(python,expected):
    code="""import importlib.metadata as m,json,sys
names=json.loads(sys.argv[1]);versions={}
for name in names:
 try:versions[name]=m.version(name)
 except m.PackageNotFoundError:versions[name]=None
print(json.dumps({'python':sys.version.split()[0],'packages':versions}))
"""
    return json.loads(subprocess.check_output([str(python),'-c',code,json.dumps(list(expected))],text=True))


def verify_install(runtime):
    expected=pins(ROOT/'requirements-f.lock')
    result=installed_versions(runtime/'f-venv/bin/python',expected)
    if result['python']!='3.11.17':raise ValueError('F runtime Python differs from 3.11.17')
    mismatches={name:(version,result['packages'][name]) for name,version in expected.items()
                if result['packages'][name]!=version}
    if mismatches:raise ValueError(f'F package pins differ: {mismatches}')
    result['model']=verify(runtime)
    result['requirements_sha256']=sha(ROOT/'requirements-f.lock')
    # Import only, without allocating a CUDA model or reading user audio.
    env=dict(os.environ,PYTHONPATH=str(ROOT/'src'),HF_HUB_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1')
    code="import sys;sys.path.insert(0,sys.argv[1]);from tf_restormer import SEInference"
    run(runtime/'f-venv/bin/python','-c',code,runtime/'vendor/TF_Restormer',env=env)
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--verify-only',action='store_true')
    ap.add_argument('--offline',action='store_true',help='Use cached packages/code/weights only')
    ap.add_argument('--with-conservative',action='store_true',help='Install the optional adaptive-mode denoiser')
    args=ap.parse_args()
    if platform.system()!='Linux' or platform.machine()!='x86_64':
        raise SystemExit('This frozen runtime targets Linux x86_64')
    runtime=Path(os.environ.get('VOICE_LAB_RUNTIME',Path.home()/'.local/share/folio')).expanduser().resolve()
    if args.verify_only:
        perform(runtime,args)
        return
    runtime.mkdir(parents=True,exist_ok=True)
    with (runtime/'.voice-lab-f-setup.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('F runtime setup is already running')
        perform(runtime,args)


def perform(runtime,args):
    venv=runtime/'f-venv';owner=venv/'.voice-lab-f-managed.json'
    if not args.verify_only:
        if venv.exists() and (not owner.is_file() or json.loads(owner.read_text()).get('managed_by')!=OWNER):
            raise ValueError('Unrecognized existing F environment preserved; use --verify-only or another VOICE_LAB_RUNTIME')
        uv=ensure_uv(runtime,args.offline)
        if not venv.exists():
            command=[uv,'venv','--python','3.11.17',venv]
            if args.offline:command.append('--offline')
            run(*command)
            owner.write_text(json.dumps({'managed_by':OWNER})+'\n')
        (venv/'.voice-lab-f-verified.json').unlink(missing_ok=True)
        lock=json.loads((ROOT/'resources/f-model-lock.json').read_text())
        ensure_source(runtime,lock,args.offline)
        for name,spec in lock['artifacts'].items():acquire_artifact(runtime,name,spec,lock,args.offline)
        command=[uv,'pip','install','--no-deps','--python',venv/'bin/python',
                 '--index-strategy','unsafe-best-match','--extra-index-url','https://download.pytorch.org/whl/cu121',
                 '-r',ROOT/'requirements-f.lock']
        if args.offline:command.append('--offline')
        run(*command,env=dict(os.environ,UV_HTTP_TIMEOUT='240'))
    result=verify_install(runtime)
    if getattr(args,'with_conservative',False):
        from voicelab.conservative import binary,install
        path=binary(runtime) if args.verify_only else install(runtime,args.offline)
        result['conservative_binary_sha256']=sha(path)
    if not args.verify_only:
        result['managed_by']=OWNER
        launcher=ROOT/'bin/voice-lab-f'
        if launcher.is_file():launcher.chmod(0o755)
        (venv/'.voice-lab-f-verified.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()

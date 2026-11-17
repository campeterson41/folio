#!/usr/bin/env python3
"""Build the pinned, unmodified master_me LADSPA engine for Folio (Linux)."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',type=Path,help='Reuse a checkout of the exact pinned commit')
    ap.add_argument('--runtime',type=Path,default=Path(os.environ.get('VOICE_LAB_RUNTIME',Path.home()/'.local/share/folio')))
    ap.add_argument('--verify-only',action='store_true')
    ap.add_argument('--offline',action='store_true',help='Verify the installed engine without network access')
    args=ap.parse_args()
    if args.verify_only or args.offline:
        sys.path.insert(0,str(ROOT/'src'))
        from voicelab.master_me import engine, verify_host
        plugin,_=engine(args.runtime.resolve());verify_host();print(plugin);return
    lock=json.loads((ROOT/'resources/mastering-lock.json').read_text())
    runtime=args.runtime.resolve();runtime.mkdir(parents=True,exist_ok=True)
    source=args.source or runtime/'vendor'/('master_me-'+lock['version'])
    if not source.exists():
        source.parent.mkdir(parents=True,exist_ok=True)
        subprocess.run(['git','clone','--recursive','--depth','1','--branch',lock['version'],lock['repository'],str(source)],check=True)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()
    if commit!=lock['commit']:
        raise ValueError('master_me checkout does not match the pinned release')
    if subprocess.check_output(['git','diff','--name-only','HEAD'],cwd=source,text=True).strip():
        raise ValueError('master_me tracked source has local modifications')
    subprocess.run(['git','submodule','update','--init','--recursive'],cwd=source,check=True)
    if any(line.startswith(('+','-','U')) for line in subprocess.check_output(
            ['git','submodule','status','--recursive'],cwd=source,text=True).splitlines()):
        raise ValueError('Unverified master_me submodules')
    subprocess.run(['make','gen'],cwd=source,check=True)
    subprocess.run(['make','-C','plugin','ladspa','-j2'],cwd=source,check=True)
    output=runtime/'mastering';output.mkdir(exist_ok=True)
    plugin=source/'bin'/lock['plugin']
    metadata=dict(lock,sha256=sha(plugin),source_directory=str(source.resolve()),
        submodules=subprocess.check_output(['git','submodule','status','--recursive'],cwd=source,text=True).splitlines())
    with tempfile.TemporaryDirectory(prefix='.install-',dir=output) as tmp:
        stage=Path(tmp);shutil.copy2(plugin,stage/lock['plugin'])
        shutil.copy2(source/'LICENSE',stage/'LICENSE')
        (stage/'manifest.json').write_text(json.dumps(metadata,indent=2)+'\n')
        # Publish the identity receipt last; an interrupted update fails closed.
        for name in [lock['plugin'],'LICENSE','manifest.json']:
            os.replace(stage/name,output/name)
    print(output/lock['plugin'])


if __name__=='__main__':main()

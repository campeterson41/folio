#!/usr/bin/env python3
"""Install/verify the pinned local Folio Studio runtime (Linux x86_64)."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from setup_f_runtime import ensure_uv
from voicelab.f_artifacts import runtime_root


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--offline',action='store_true',help='Use verified local caches only')
    ap.add_argument('--verify-only',action='store_true')
    args=ap.parse_args()
    runtime=runtime_root()
    runtime.mkdir(parents=True,exist_ok=True)
    with (runtime/'.folio-studio-setup.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Studio runtime setup is already running')
        perform(args)


def perform(args):
    command=[sys.executable,str(ROOT/'tools/setup_f_runtime.py'),'--with-conservative']
    if args.offline:command.append('--offline')
    if args.verify_only:command.append('--verify-only')
    subprocess.run(command,check=True)
    runtime=runtime_root();python=runtime/'f-venv/bin/python'
    if not args.verify_only:
        uv=ensure_uv(runtime,args.offline)
        install=[str(uv),'pip','install','--no-deps','--python',str(python),'-r',str(ROOT/'requirements-studio.txt')]
        if args.offline:install.append('--offline')
        subprocess.run(install,check=True)
    code="""import importlib.metadata as m,json
from nara_wpe.wpe import wpe_v8
expected={'nara-wpe':'0.0.11','click':'8.5.0'}
actual={k:m.version(k) for k in expected}
assert actual==expected,(actual,expected)
print(json.dumps(actual))
"""
    result=subprocess.check_output([str(python),'-c',code],env=dict(os.environ,HF_HUB_OFFLINE='1'),text=True)
    print('Folio Studio verified: '+result.strip())


if __name__=='__main__':main()

"""Verify the exact public TF checkpoint used for the selected F sound."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

LOCK=Path(__file__).resolve().parents[2]/'resources/f-model-lock.json'


def runtime_root():
    return Path(os.environ.get('VOICE_LAB_RUNTIME',
        Path.home()/'.local/share/folio')).expanduser().resolve()


def verify(runtime,lock_path=LOCK,verify_source=True):
    runtime=Path(runtime);lock=json.loads(Path(lock_path).read_text())
    verified={}
    for name,artifact in lock['artifacts'].items():
        path=runtime/artifact['path']
        with path.open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
        if actual!=artifact['sha256']:
            raise ValueError(f'F model artifact differs from the selected checkpoint: {name}')
        verified[name]=actual
    if verify_source:
        vendor=runtime/lock['source']['directory']
        revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=vendor,text=True).strip()
        if revision!=lock['source']['revision']:
            raise ValueError('F model source revision differs from the selected implementation')
        subprocess.run(['git','diff','--exit-code','--quiet','HEAD'],cwd=vendor,check=True)
    return {'model_revision':lock['model_revision'],
            'source_revision':lock['source']['revision'],'artifacts_sha256':verified}

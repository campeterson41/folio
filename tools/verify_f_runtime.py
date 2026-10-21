"""Read-only verification of the selected local F source, weights and packages."""
import importlib.metadata
import json
import os
from pathlib import Path
import sys
from voicelab.f_artifacts import verify

runtime=Path(os.environ.get('VOICE_LAB_RUNTIME',Path.home()/'.local/share/folio'))
result=verify(runtime)
result['python']=sys.version.split()[0]
result['packages']={p:importlib.metadata.version(p) for p in
    ['torch','numpy','scipy','soundfile','loguru','rotary-embedding-torch','einops','PyYAML']}
print(json.dumps(result,indent=2))

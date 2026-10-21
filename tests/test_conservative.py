import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest
import soundfile as sf
from voicelab import conservative
from voicelab.streaming import sha

def test_installer_can_import_backend_without_site_packages():
    source=Path(__file__).resolve().parents[1]/'src'
    subprocess.run([sys.executable,'-S','-c',
        'import sys;sys.path.insert(0,sys.argv[1]);import voicelab.conservative',str(source)],check=True)

def test_unknown_binary_is_preserved_and_rejected(tmp_path):
    spec=json.loads(conservative.LOCK.read_text());p=tmp_path/spec['path'];p.parent.mkdir();p.write_bytes(b'unknown')
    with pytest.raises(ValueError):conservative.install(tmp_path,offline=True)
    assert p.read_bytes()==b'unknown'

def test_cache_cannot_refer_to_another_recording(tmp_path):
    p=tmp_path/'cache.wav';sf.write(p,np.zeros(4800),48000,subtype='PCM_24')
    m={'recipe':'deepfilternet3-conservative-native','source_sha256':'sourceA','output_sha256':sha(p),
       'attenuation_limit_db':12,'delay_compensated':True}
    p.with_suffix('.json').write_text(json.dumps(m))
    assert conservative.validate(p,'sourceA',4800)==m
    with pytest.raises(ValueError):conservative.validate(p,'sourceB',4800)
    m['attenuation_limit_db']=100;p.with_suffix('.json').write_text(json.dumps(m))
    with pytest.raises(ValueError):conservative.validate(p,'sourceA',4800)

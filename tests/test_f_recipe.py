"""Cache provenance must prevent a different recording entering the processor."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import argparse
import fcntl
import shutil

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from enhance_f import validate_native, sha
import enhance_f


def test_cached_restoration_requires_matching_source_content_and_recipe(tmp_path):
    stem = tmp_path/'native.wav'
    sf.write(stem,np.zeros(4800),48000,subtype='PCM_24')
    receipt = {'source_sha256':'expected', 'output_sha256':sha(stem),
               'family':'tf-restormer', 'parameters':{'seconds':3,'input_rate':48000}}
    meta=stem.with_suffix('.json')
    meta.write_text(json.dumps(receipt))
    assert validate_native(stem,'expected',4800)==receipt
    with pytest.raises(ValueError):validate_native(stem,'other-source',4800)
    with pytest.raises(ValueError):validate_native(stem,'expected',9600)
    receipt['parameters']['seconds']=4
    meta.write_text(json.dumps(receipt))
    with pytest.raises(ValueError):validate_native(stem,'expected',4800)
    receipt['parameters']['seconds']=3
    meta.write_text(json.dumps(receipt))
    sf.write(stem,np.ones(4800)*.01,48000,subtype='PCM_24')
    with pytest.raises(ValueError):validate_native(stem,'expected',4800)


def test_changed_source_is_not_published_with_an_old_hash(tmp_path,monkeypatch):
    source=tmp_path/'source.wav';stem=tmp_path/'native.wav';out=tmp_path/'out.wav'
    sf.write(source,np.ones(4800)*.01,48000,subtype='PCM_24')
    shutil.copyfile(source,stem)
    preset=tmp_path/'preset.json';preset.write_text('{}')
    monkeypatch.setattr(enhance_f,'validate_native',lambda *args:{})
    def master(inp,target,preset):
        shutil.copyfile(inp,target)
        with source.open('ab') as f:f.write(b'changed during processing')
        return {}
    def preserve(inp,wet,target,**kwargs):
        shutil.copyfile(wet,target)
        return {}
    monkeypatch.setattr(enhance_f,'master_file',master)
    monkeypatch.setattr(enhance_f,'preserve_file',preserve)
    args=argparse.Namespace(input=source,output=out,preset=preset,
                            native_stem=stem,device='cpu',preserve_voiced_fricatives=False)
    with pytest.raises(RuntimeError,match='Source changed'):
        enhance_f.process(args)
    assert not out.exists() and not out.with_suffix('.json').exists()
    assert not list(tmp_path.glob('voice-lab-f-*'))


def test_cli_reserves_output_before_starting_work(tmp_path,monkeypatch):
    target=tmp_path/'out.wav'
    monkeypatch.setattr(sys,'argv',['enhance_f','--input',str(tmp_path/'missing.wav'),
                                   '--output',str(target)])
    with (tmp_path/'.out.json.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(FileExistsError,match='already being processed'):
            enhance_f.main()

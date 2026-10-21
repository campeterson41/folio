import hashlib
import json
from pathlib import Path
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import setup_f_runtime as setup


def test_dependencies_must_be_exactly_pinned(tmp_path):
    lock=tmp_path/'deps.lock';lock.write_text('# fixed\nnumpy==1.26.4\ntorch==2.5.1+cu121\n')
    assert setup.pins(lock)=={'numpy':'1.26.4','torch':'2.5.1+cu121'}
    lock.write_text('numpy>=1.26\n')
    with pytest.raises(ValueError):setup.pins(lock)


def test_existing_unrecognized_model_is_preserved(tmp_path):
    target=tmp_path/'model.pt';target.write_bytes(b'preserve this model')
    spec={'path':'model.pt','sha256':'0'*64}
    with pytest.raises(ValueError,match='preserved'):
        setup.acquire_artifact(tmp_path,'model.pt',spec,{})
    assert target.read_bytes()==b'preserve this model'


def test_offline_setup_does_not_contact_network_for_missing_weights(tmp_path,monkeypatch):
    home=tmp_path/'home';home.mkdir()
    monkeypatch.setattr(setup.Path,'home',lambda:home)
    monkeypatch.setattr(setup.urllib.request,'urlopen',lambda *a,**k:pytest.fail('network attempted'))
    spec={'path':'models/model.pt','sha256':hashlib.sha256(b'weights').hexdigest()}
    with pytest.raises(FileNotFoundError,match='unavailable locally'):
        setup.acquire_artifact(tmp_path/'runtime','model.pt',spec,{},offline=True)
    assert not (tmp_path/'runtime/models/model.pt').exists()


def test_unowned_environment_is_not_modified(tmp_path):
    runtime=tmp_path/'runtime';venv=runtime/'f-venv';venv.mkdir(parents=True)
    sentinel=venv/'keep';sentinel.write_text('unrelated environment')
    import argparse
    with pytest.raises(ValueError,match='preserved'):
        setup.perform(runtime,argparse.Namespace(verify_only=False,offline=True))
    assert sentinel.read_text()=='unrelated environment'

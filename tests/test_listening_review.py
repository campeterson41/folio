import base64
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.error
import urllib.request

import numpy as np
import pytest
import soundfile as sf

from voicelab.listening_review import prepare, review, select_windows, validate


@pytest.fixture
def recordings(tmp_path):
    t = np.arange(48000 * 5) / 48000
    x = .02 * np.sin(2 * np.pi * 120 * t)
    paths = [tmp_path / f'{n}.wav' for n in range(3)]
    for p, gain in zip(paths, [1, .2, 3]):
        sf.write(p, x * gain, 48000, subtype='PCM_24')
    return paths


def response(ids=('excerpt-1',)):
    return {'comparisons': [dict(id=i, preferred='uncertain', confidence=.2,
        observations='No reliable audible preference.', bass_db=0, presence_db=0,
        restoration='uncertain') for i in ids]}


def test_selected_excerpts_only_and_level_matching(recordings):
    before = [p.read_bytes() for p in recordings]
    parts, receipt = prepare(recordings, [(1, 2)])
    clips = [sf.read(io.BytesIO(base64.b64decode(p['inlineData']['data'])))[0]
             for p in parts if 'inlineData' in p]
    assert len(clips) == 3
    assert all(len(x) == 96000 for x in clips)
    rms = [np.sqrt(np.mean(x ** 2)) for x in clips]
    assert max(rms) - min(rms) < 1e-5
    assert receipt['uploaded_audio_seconds'] == 6
    assert before == [p.read_bytes() for p in recordings]
    assert set(receipt['version_mapping']) == {'raw', 'A', 'B'}


@pytest.mark.parametrize('windows', [[(0, 16)], [(-1, 1)], [(4, 2)], [(0, 1)] * 4,
                                      [(float('nan'), 1)], [(0, float('inf'))]])
def test_upload_bounds_fail_before_network(recordings, windows):
    with pytest.raises(ValueError):
        review(recordings, windows, send=True, opener=lambda *a, **k: pytest.fail('Network called'))


def test_dry_run_never_uses_credentials_or_network(recordings, monkeypatch):
    monkeypatch.setenv('GEMINI_API_KEY', 'test-only-not-a-real-key')
    result = review(recordings, [(0, 1)], opener=lambda *a, **k: pytest.fail('Network called'))
    assert result['status'] == 'prepared-not-sent'
    assert not result['recommendations_applied']
    assert 'test-only' not in json.dumps(result)


def test_missing_credentials_do_not_send(recordings, monkeypatch):
    monkeypatch.delenv('GEMINI_API_KEY', raising=False)
    monkeypatch.delenv('GOOGLE_API_KEY', raising=False)
    with pytest.raises(ValueError, match='no audio sent'):
        review(recordings, [(0, 1)], send=True, opener=lambda *a, **k: pytest.fail('Network called'))


def test_structured_request_and_usage_receipt(recordings, monkeypatch):
    monkeypatch.setenv('GEMINI_API_KEY', 'test-only-not-a-real-key')
    calls = []
    def send(request, timeout):
        calls.append(request)
        assert timeout == 120
        assert request.full_url.startswith('https://generativelanguage.googleapis.com/v1beta/models/')
        assert 'test-only' not in request.full_url
        body = json.loads(request.data)
        assert body['generationConfig']['maxOutputTokens'] == 1800
        assert len([p for p in body['contents'][0]['parts'] if 'inlineData' in p]) == 3
        return io.BytesIO(json.dumps({'candidates': [{'finishReason': 'STOP',
            'content': {'parts': [{'text': json.dumps(response())}]}}],
            'usageMetadata': {'promptTokenCount': 200}, 'modelVersion': 'test-model'}).encode())
    result = review(recordings, [(0, 1)], send=True, opener=send)
    assert len(calls) == 1
    assert result['status'] == 'reviewed'
    assert result['usage']['promptTokenCount'] == 200
    assert 'test-only' not in json.dumps(result)
    assert not result['recommendations_applied']


@pytest.mark.parametrize('change', [dict(bass_db=4), dict(presence_db=float('nan')),
    dict(confidence=True), dict(restoration='run shell'), dict(id='unrequested'), dict(extra='command')])
def test_reviewer_cannot_supply_unbounded_actions(change):
    value = response()
    value['comparisons'][0].update(change)
    with pytest.raises(ValueError):
        validate(value, ['excerpt-1'])


def test_diverse_automatic_selection_stays_inside_recording():
    rows = [dict(start_s=t, end_s=t+.5, pitch_hz=200 if 20<t<30 else 100,
                 conservative_fraction=1 if 20<t<30 else 0, source_level_dbfs=-30,
                 loud_outlier_fraction=1 if t==65 else 0) for t in np.arange(0, 80, .5)]
    windows = select_windows({'adaptive_control': {'timeline': rows}}, 80)
    assert len(windows) == 3
    assert all(0 <= a < a+n <= 80 and n <= 15 for a, n in windows)
    assert any(a <= 25 < a+n for a, n in windows)
    assert any(a <= 65 < a+n for a, n in windows)


def test_incomplete_api_response_is_not_accepted(recordings, monkeypatch):
    monkeypatch.setenv('GEMINI_API_KEY', 'test-only-not-a-real-key')
    def send(*a, **k):
        return io.BytesIO(b'{"candidates":[{"finishReason":"MAX_TOKENS"}]}')
    result = review(recordings, [(0, 1)], send=True, opener=send)
    assert result['status'] == 'response-rejected'
    assert 'review' not in result


@pytest.mark.parametrize('payload', [[], {'candidates': ['x']},
    {'candidates': [{'finishReason': 'STOP', 'content': {'parts': [{'text': 5}]}}]},
    {'candidates': [{'finishReason': 'STOP', 'content': {'parts': [{'text': '{"comparisons":NaN}'}]}}]}])
def test_malformed_provider_responses_retain_attempt(recordings, monkeypatch, payload):
    monkeypatch.setenv('GEMINI_API_KEY', 'test-only-not-a-real-key')
    checkpoints = []
    result = review(recordings, [(0, 1)], send=True,
        opener=lambda *a, **k: io.BytesIO(json.dumps(payload).encode()),
        recorder=lambda r: checkpoints.append(dict(r)))
    assert checkpoints[0]['status'] == 'upload-attempted'
    assert 'upload_attempted_at' in result
    assert result['status'] == 'response-rejected'
    assert 'review' not in result


def test_http_failure_has_receipt_without_body_or_credentials(recordings, monkeypatch):
    monkeypatch.setenv('GEMINI_API_KEY', 'test-only-not-a-real-key')
    calls = []
    def send(*a, **k):
        calls.append(1)
        raise urllib.error.HTTPError('https://example.invalid', 429, 'private message', {},
                                     io.BytesIO(b'private error body'))
    result = review(recordings, [(0, 1)], send=True, opener=send)
    assert len(calls) == 1
    assert result['status'] == 'upload-attempted-failed'
    assert result['error_code'] == 'http-429'
    assert 'private' not in json.dumps(result)
    assert 'test-only' not in json.dumps(result)


def test_advice_names_actual_blinded_target(recordings, monkeypatch):
    monkeypatch.setenv('GEMINI_API_KEY', 'test-only-not-a-real-key')
    result = review(recordings, [(0, 1)])
    assert result['advice_applies_to'] == ['raw', 'current', 'adaptive'][result['version_mapping']['B']]


def test_broader_google_key_is_never_an_implicit_fallback(recordings, monkeypatch):
    monkeypatch.delenv('GEMINI_API_KEY', raising=False)
    monkeypatch.setenv('GOOGLE_API_KEY', 'test-only-not-a-real-key')
    with pytest.raises(ValueError, match='no audio sent'):
        review(recordings, [(0, 1)], send=True, opener=lambda *a, **k: pytest.fail('Called'))


def test_credential_header_is_not_forwarded_on_redirect():
    from voicelab.listening_review import _NoRedirect
    request = urllib.request.Request('https://generativelanguage.googleapis.com',
                                     headers={'x-goog-api-key': 'test-only'})
    assert _NoRedirect().redirect_request(request, None, 302, 'redirect', {},
                                          'https://example.invalid') is None


def test_cli_retains_preflight_failure_and_preserves_existing_output(recordings, tmp_path):
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / 'attempt.json'
    env = dict(os.environ, PYTHONPATH=str(root / 'src'))
    env.pop('GEMINI_API_KEY', None)
    command = [sys.executable, str(root/'tools/review_audio.py'), '--raw', str(recordings[0]),
               '--current', str(recordings[1]), '--adaptive', str(recordings[2]),
               '--excerpt', '0:1', '--output', str(output), '--send']
    p = subprocess.run(command, env=env, capture_output=True, text=True)
    assert p.returncode == 2
    assert json.loads(output.read_text())['status'] == 'not-sent-preflight-failed'
    before = output.read_bytes()
    p = subprocess.run(command, env=env, capture_output=True, text=True)
    assert p.returncode == 2
    assert output.read_bytes() == before

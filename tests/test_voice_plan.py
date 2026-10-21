import json
import numpy as np
import pytest
import soundfile as sf
from voicelab.adaptive import analyze
from voicelab.listening_review import _hash
from voicelab.voice_plan import utterances, policy, apply


def fixture(tmp_path, shift=0):
    t = np.arange(48000*10)/48000
    high = (t > 3.7+shift) & (t < 5.5+shift)
    active = ((t > .7)&(t < 3.2)) | high | ((t > 6.7)&(t < 9.5))
    f = np.where(high, 230, 110)
    phase = np.cumsum(2*np.pi*f/48000)
    x = .04*(np.sin(phase)+.3*np.sin(phase*2))*active
    src, base, gentle = [tmp_path/(n+'.wav') for n in ['raw','base','gentle']]
    y = x*.5; y[high] *= .15
    for p, z in [(src,x),(base,y),(gentle,x)]:
        sf.write(p,z,48000,subtype='PCM_24')
    receipt = dict(status='reviewed',model='gemini-test', source_sha256=[_hash(src),'comparison',_hash(base)],
                   version_mapping={'raw':0,'A':1,'B':2},
                   excerpts=[dict(id='character',start_seconds=3.5+shift,duration_seconds=2.5),
                             dict(id='normal',start_seconds=.8,duration_seconds=2)],
                   review={'comparisons':[
                       dict(id='character',preferred='uncertain',confidence=.5,observations='Gentler.',
                            bass_db=0,presence_db=.1,restoration='gentler'),
                       dict(id='normal',preferred='B',confidence=.6,observations='Keep normal.',
                            bass_db=0,presence_db=0,restoration='keep')]})
    report=tmp_path/'review.json';report.write_text(json.dumps(receipt))
    return src,base,gentle,report


@pytest.mark.parametrize('shift',[0,.6])
def test_whole_utterance_and_pause_boundaries_move_with_voice(tmp_path,shift):
    source,*_=fixture(tmp_path,shift)
    features=analyze(source);decision=utterances(features)
    assert len(decision['regions'])==1
    a,b=decision['regions'][0]
    assert a/48000<3.7+shift and b/48000>5.5+shift
    assert not (features['level_dbfs'][max(0,a//960):a//960+4]>decision['speech_threshold_dbfs']).any()
    assert b-a>48000


def test_applied_policy_preserves_regular_voice_and_repairs_bounded_character(tmp_path):
    src,base,gentle,report=fixture(tmp_path)
    inputs=[p.read_bytes() for p in (src,base,gentle,report)]
    output=tmp_path/'out.wav';meta=apply(src,base,gentle,report,output,chunk_seconds=2)
    y,_=sf.read(output);old,_=sf.read(base)
    assert len(y)==len(old)
    np.testing.assert_array_equal(y[:48000*3],old[:48000*3])
    np.testing.assert_array_equal(y[48000*7:],old[48000*7:])
    assert np.sqrt(np.mean(y[48000*4:48000*5]**2))>np.sqrt(np.mean(old[48000*4:48000*5]**2))*3
    assert meta['plan']['normal_narration']=='keep'
    assert meta['plan']['presence_db']==.1
    assert inputs==[p.read_bytes() for p in (src,base,gentle,report)]
    with pytest.raises(FileExistsError):apply(src,base,gentle,report,output)


def test_wrong_recording_or_unreviewed_advice_is_rejected(tmp_path):
    src,base,gentle,report=fixture(tmp_path)
    features=analyze(src);decision=utterances(features)
    data=json.loads(report.read_text());data['source_sha256'][0]='other'
    with pytest.raises(ValueError,match='raw/base pair'):
        policy(data,features,decision,_hash(src),_hash(base))
    data=json.loads(report.read_text());data['status']='prepared-not-sent'
    with pytest.raises(ValueError,match='completed Gemini'):
        policy(data,features,decision,_hash(src),_hash(base))


def test_uncertain_review_cannot_be_presented_as_a_new_ai_policy(tmp_path):
    src,base,gentle,report=fixture(tmp_path)
    data=json.loads(report.read_text());data['review']['comparisons'][0]['restoration']='uncertain'
    features=analyze(src)
    with pytest.raises(ValueError,match='does not support'):
        policy(data,features,utterances(features),_hash(src),_hash(base))


def test_loud_mic_test_does_not_trigger_a_restoration_path_switch():
    n=500
    features=dict(level_dbfs=np.full(n,-40.), pitch_hz=np.full(n,110.),
                  periodicity=np.full(n,.8), samples=n*960)
    features['level_dbfs'][100:120]=-5
    features['pitch_hz'][100:120]=250
    assert utterances(features)['regions']==[]


def test_float_base_is_rejected_instead_of_claiming_exact_preservation(tmp_path):
    src,base,gentle,report=fixture(tmp_path)
    x,rate=sf.read(base);sf.write(base,x,rate,subtype='FLOAT')
    with pytest.raises(ValueError,match='Base must be PCM16/24'):
        apply(src,base,gentle,report,tmp_path/'out.wav')

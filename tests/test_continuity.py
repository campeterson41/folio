import numpy as np
import pytest
import soundfile as sf
from voicelab.continuity import choose_mix


def paths(tmp_path, transform):
    t=np.arange(48000*4)/48000
    base=.04*np.sin(2*np.pi*180*t)
    candidate=transform(base.copy())
    wet,dry=tmp_path/'base.wav',tmp_path/'candidate.wav'
    sf.write(wet,base,48000,subtype='PCM_24')
    sf.write(dry,candidate,48000,subtype='PCM_24')
    return wet,dry


def test_candidate_that_erases_a_quiet_syllable_keeps_base_underneath(tmp_path):
    def erase(x):
        x[48000*2:48000*2+2400]=0
        return x
    wet,dry=paths(tmp_path,erase)
    result=choose_mix(wet,dry,[[9600,182400]],0,0,audible_floor_dbfs=-65,chunk_seconds=1)
    row=result['regions'][0]
    assert row['mix']==.1
    assert min(row['proposed_min_change_db']) < -50
    assert min(row['applied_min_change_db'])>=-1
    # The one weight applies to the COMPLETE utterance, never just its bad frame.
    other=choose_mix(wet,dry,[[9600,182400]],0,0,audible_floor_dbfs=-65,chunk_seconds=3)
    assert other['regions'][0]['mix']==row['mix']


def test_equal_level_opposite_phase_is_not_safe_to_crossfade(tmp_path):
    wet,dry=paths(tmp_path,lambda x:-x)
    result=choose_mix(wet,dry,[[9600,182400]],0,0,audible_floor_dbfs=-65)
    assert result['regions'][0]['mix']==0


def test_loudness_pulses_are_bounded_as_well_as_dropouts(tmp_path):
    wet,dry=paths(tmp_path,lambda x:x*3)
    row=choose_mix(wet,dry,[[9600,182400]],0,0,audible_floor_dbfs=-65)['regions'][0]
    assert 0<row['mix']<1
    assert max(row['applied_max_change_db'])<=3


def test_unchanged_candidate_is_not_rejected(tmp_path):
    wet,dry=paths(tmp_path,lambda x:x)
    row=choose_mix(wet,dry,[[9600,182400]],0,0)['regions'][0]
    assert row['mix']==1


def test_invalid_continuity_limits_fail(tmp_path):
    wet,dry=paths(tmp_path,lambda x:x)
    with pytest.raises(ValueError):
        choose_mix(wet,dry,[[0,48000]],0,0,maximum_loss_db=float('nan'))

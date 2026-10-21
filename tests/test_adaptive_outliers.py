import numpy as np
from voicelab.adaptive import controller

def test_a_loud_mic_test_is_not_eq_calibration_speech():
    n=500
    features={'level_dbfs':np.full(n,-40.),'pitch_hz':np.full(n,110.),
        'periodicity':np.full(n,.8),'samples':n*960}
    features['level_dbfs'][100:120]=-5.
    features['pitch_hz'][100:120]=240.
    result=controller(features)
    assert result['overload_frames'][100:120].all()
    assert not result['selected'][100:120].any()
    assert not result['overload_frames'][:50].any()

import numpy as np
from voicelab.chunks import restore_chunks


def test_overlap_add_has_unity_gain_and_complete_tail():
    x = np.random.default_rng(3).normal(size=19713)
    y = restore_chunks(x, 1000, lambda x: x)
    assert len(y) == len(x)
    np.testing.assert_allclose(x, y, atol=1e-12)


def test_overlap_add_smooths_context_dependent_chunk_offsets():
    x = np.arange(10000, dtype=float) / 10000
    # A context-dependent error is an abrupt discontinuity without a crossfade.
    y = restore_chunks(x, 1000, lambda z: z + z[0] * 0.1)
    assert np.max(abs(np.diff(y))) < 0.001


def test_upsampler_clock_and_boundaries_are_preserved():
    x = np.arange(17009, dtype=float)
    y = restore_chunks(x, 1000, lambda z: np.repeat(z, 3), output_factor=3)
    np.testing.assert_allclose(y, np.repeat(x, 3), atol=1e-10)

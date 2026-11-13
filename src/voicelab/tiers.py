"""Speed/quality tiers for Folio Studio.

``reference`` is the approved recipe: the same arithmetic as v0.3, only
scheduled better. The other tiers trade measured distance from that render for
time: they give the restoration model less bandwidth to process, evaluate it in
slices (and float16 on CUDA), run the denoiser on parallel segments, and leave
model windows without speech silent.
"""
from __future__ import annotations

TIERS = {
    'reference': {
        'summary': 'Approved v0.3 sound. Same arithmetic, computed in parallel.',
        'model_rate': 48000, 'forward': {'cpu': 'lean', 'cuda': 'stock'},
        'precision': {'cpu': 'float32', 'cuda': 'float32'}, 'threads': 4,
        'denoise_segments': False, 'skip_silence': False},
    'high': {
        'summary': 'Full-band model input, faster evaluation; about 77 dB from the reference render.',
        'model_rate': 48000, 'forward': {'cpu': 'sliced', 'cuda': 'sliced'},
        'precision': {'cpu': 'float32', 'cuda': 'float16'},
        'denoise_segments': True, 'skip_silence': True},
    'balanced': {
        'summary': 'Model hears 12 kHz and regenerates the rest; within 1 dB per octave of reference.',
        'model_rate': 24000, 'forward': {'cpu': 'sliced', 'cuda': 'sliced'},
        'precision': {'cpu': 'float32', 'cuda': 'float16'},
        'denoise_segments': True, 'skip_silence': True},
    'fast': {
        'summary': 'Model hears 8 kHz, its training condition; top octave 3-5 dB quieter.',
        'model_rate': 16000, 'forward': {'cpu': 'sliced', 'cuda': 'sliced'},
        'precision': {'cpu': 'float32', 'cuda': 'float16'},
        'denoise_segments': True, 'skip_silence': True},
    'draft': {
        'summary': 'Model hears 4 kHz; for checking edits and levels, not for release.',
        'model_rate': 8000, 'forward': {'cpu': 'sliced', 'cuda': 'sliced'},
        'precision': {'cpu': 'float32', 'cuda': 'float16'},
        'denoise_segments': True, 'skip_silence': True},
}
DEFAULT = 'reference'


def resolve(name, device):
    """Concrete settings of a tier on ``device`` ('cpu' or 'cuda')."""
    if name not in TIERS:
        raise ValueError(f'Unknown quality tier {name!r}; choose from {", ".join(TIERS)}')
    tier = TIERS[name]
    return {'tier': name, 'model_rate': tier['model_rate'], 'forward': tier['forward'][device],
            'precision': tier['precision'][device], 'threads': tier.get('threads'),
            'denoise_segments': tier['denoise_segments'], 'skip_silence': tier['skip_silence']}


def stem_matches(parameters, settings):
    """Whether a saved native stem was made with settings this tier accepts.

    Stems from v0.3 carry none of these fields and count as the reference.
    """
    forward = parameters.get('forward', 'stock')
    made = (parameters.get('model_input_rate', 48000), parameters.get('precision', 'float32'),
            bool(parameters.get('skipped_windows', 0)))
    wanted = (settings['model_rate'], settings['precision'], False)
    if settings['tier'] == 'reference':
        return made == wanted and forward in ('stock', 'lean')
    return made[:2] == wanted[:2] and (settings['skip_silence'] or not made[2])

"""Apply bounded Gemini advice through source-derived utterance protection.

Gemini proposes a coarse policy. It does not supply waveform samples, repair
words, or select the output's word locations. No reference audio is read here.
"""
from pathlib import Path
import json
import tempfile
import numpy as np
import soundfile as sf
from scipy import ndimage, signal
from .adaptive import analyze, _read, HOP
from .consonant_plan import source_profile
from .listening_review import _hash, validate
from .streaming import stage
from . import dsp
from .continuity import choose_mix, fade


def utterances(features):
    level = features['level_dbfs']
    threshold = max(float(np.percentile(level, 15)) + 18,
                    float(np.percentile(level, 90)) - 28)
    if np.percentile(level, 90) - np.percentile(level, 15) < 18:
        threshold = float(np.percentile(level, 90)) - 25
    active = level > threshold
    voiced = active & (features['periodicity'] > .6)
    overload = level >= float(np.percentile(level, 90)) + 12
    baseline = float(np.percentile(features['pitch_hz'][voiced], 40)) if voiced.sum() >= 10 else None
    high = voiced & ~overload & (features['pitch_hz'] > (baseline or 500) * 1.35)
    high = ndimage.binary_opening(high, structure=np.ones(3))
    # A brief pitch excursion cannot change paths within a word.
    joined = ndimage.binary_closing(active, structure=np.ones(15))
    labels, _ = ndimage.label(joined)
    regions = []
    for obj in ndimage.find_objects(labels):
        span = obj[0]
        # A near-mic overload is a level-control event, not a character voice.
        # Keep its complete utterance on the existing, peak-safe path.
        if overload[span].any():
            continue
        if high[span].sum() < 3 or high[span].sum() < .15 * max(1, voiced[span].sum()):
            continue
        a = max(0, span.start * HOP - 6720)
        b = min(features['samples'], span.stop * HOP + 6720)
        if b - a < 24000:
            continue
        if regions and a <= regions[-1][1]:
            regions[-1][1] = max(b, regions[-1][1])
        else:
            regions.append([a, b])
    trusted = voiced & ~high & ~overload
    return {'baseline_pitch_hz': baseline, 'speech_threshold_dbfs': threshold,
            'voiced': voiced, 'high': high, 'trusted': trusted, 'regions': regions}


def policy(receipt, features, decision, source_hash, base_hash):
    if receipt.get('status') != 'reviewed' or not str(receipt.get('model', '')).startswith('gemini-'):
        raise ValueError('A completed Gemini audio review is required')
    hashes = receipt.get('source_sha256', [])
    mapping = receipt.get('version_mapping', {})
    index = mapping.get('B')
    if (type(index) is not int or index not in (1, 2) or len(hashes) != 3 or
            hashes[0] != source_hash or hashes[index] != base_hash):
        raise ValueError('Review advice does not refer to this raw/base pair')
    excerpts = receipt.get('excerpts', [])
    ids = [r['id'] for r in excerpts]
    rows = validate(receipt['review'], ids)['comparisons']
    by_id = {r['id']: r for r in rows}
    high_advice = []
    classifications = []
    for entry in excerpts:
        a = round(entry['start_seconds'] / .02)
        b = round((entry['start_seconds'] + entry['duration_seconds']) / .02)
        if not 0 <= a < b <= len(features['level_dbfs']):
            raise ValueError('Review excerpt outside source')
        voice = decision['voiced'][a:b]
        fraction = float(decision['high'][a:b].sum() / max(1, voice.sum()))
        row = by_id[entry['id']]
        classifications.append({'id': entry['id'], 'higher_register_fraction': fraction,
                                'restoration_advice': row['restoration']})
        if fraction >= .1 and row['restoration'] == 'gentler':
            high_advice.append(row)
    if not high_advice:
        raise ValueError('Review does not support a higher-register gentle-restoration policy')
    return {'version': 1, 'model': receipt['model'], 'higher_register_restoration': 'gentler',
            'normal_narration': 'keep',
            'bass_db': float(np.median([r['bass_db'] for r in high_advice])),
            'presence_db': float(np.median([r['presence_db'] for r in high_advice])),
            'review_classification': classifications,
            'note': 'Excerpt features classify the advice. Output regions are inferred afresh over the whole recording; excerpt/word timestamps never become patches.'}


def apply(source, base, gentle, review_receipt, target, chunk_seconds=30):
    source, base, gentle, review_receipt, target = map(Path, (source, base, gentle, review_receipt, target))
    if target.exists() or target.with_suffix('.json').exists():
        raise FileExistsError(target)
    if target.resolve() in [p.resolve() for p in (source, base, gentle, review_receipt)]:
        raise ValueError('Cannot overwrite an input')
    infos = [sf.info(p) for p in (source, base, gentle)]
    if any(i.frames != infos[0].frames or i.samplerate != 48000 or i.channels != 1 for i in infos):
        raise ValueError('Expected matching 48kHz mono WAVs')
    if infos[1].subtype not in ('PCM_16', 'PCM_24'):
        raise ValueError('Base must be PCM16/24 so unchanged samples remain exact')
    hashes = [_hash(p) for p in (source, base, gentle, review_receipt)]
    receipt = json.loads(review_receipt.read_text())
    features = analyze(source)
    decision = utterances(features)
    plan = policy(receipt, features, decision, hashes[0], hashes[1])
    trusted = decision['trusted']
    if trusted.sum() < 10:
        raise ValueError('Not enough ordinary speech to match the restoration paths')
    with sf.SoundFile(gentle) as dry, sf.SoundFile(base) as wet:
        profile = source_profile(np.repeat(trusted, 2), [], infos[0].frames,
                                 lambda a, n: _read(dry, a, n), lambda a, n: _read(wet, a, n))
    if profile:
        profile['gains_db'] = np.clip(np.array(profile['gains_db']) * .7, -4.5, 4.5).tolist()
        freq = np.array(profile['frequencies_hz'])
        # Small, broad advice is applied only to the protected restoration path.
        extra = (plan['bass_db'] / (1 + (freq / 160) ** 4) +
                 plan['presence_db'] * np.exp(-.5 * (np.log2(freq / 2800) / .8) ** 2))
        profile['gains_db'] = (np.array(profile['gains_db']) + extra).tolist()
    target.parent.mkdir(parents=True, exist_ok=True)
    regions = decision['regions']
    with tempfile.TemporaryDirectory(prefix='folio-voice-plan-', dir=target.parent) as tmp:
        tone = Path(tmp) / 'gentle-tone.wav'
        stage(gentle, tone, lambda x: dsp.apply_tone(dsp.highpass(x, 50), profile),
              lambda *a: None, 'gentle tone', chunk_seconds, 4, subtype='DOUBLE')
        dry_features, wet_features = analyze(tone), analyze(base)
        ratios = wet_features['level_dbfs'][trusted] - dry_features['level_dbfs'][trusted]
        gain_db = float(np.median(ratios))
        indexes = np.flatnonzero(trusted)
        # Match the paths with one robust gain, instead of chasing suppressed words.
        lags = []
        with sf.SoundFile(tone) as dry, sf.SoundFile(base) as wet:
            for i in indexes[np.linspace(0, len(indexes) - 1, min(24, len(indexes))).astype(int)]:
                a = int(i) * HOP
                if a + 4096 > infos[0].frames:
                    continue
                x, y = _read(dry, a, 4096), _read(wet, a, 4096)
                x -= x.mean(); y -= y.mean()
                cor = signal.correlate(y, x, method='fft')
                lag = int(np.argmax(cor[4095 - 240:4095 + 241]) - 240)
                if abs(lag) < 240:
                    lags.append(lag)
        delay = int(round(np.median(lags))) if lags else 0
        audible_floor = max(float(np.percentile(wet_features['level_dbfs'],15))+12,
                            float(np.percentile(wet_features['level_dbfs'],90))-40, -80.)
        continuity = choose_mix(base,tone,regions,gain_db,delay,audible_floor_dbfs=audible_floor,
                                chunk_seconds=chunk_seconds)
        scratch = Path(tmp) / 'result.wav'
        fmt = 'RF64' if infos[0].frames * 3 > 3900000000 else 'WAV'
        with sf.SoundFile(tone) as dry, sf.SoundFile(base) as wet, sf.SoundFile(scratch, 'x',
                samplerate=48000, channels=1, subtype='PCM_24', format=fmt) as out:
            for start in range(0, infos[0].frames, round(chunk_seconds * 48000)):
                n = min(round(chunk_seconds * 48000), infos[0].frames - start)
                x = _read(dry, start - delay, n) * 10 ** (gain_db / 20)
                wet.seek(start); y = wet.read(n, dtype='float64')
                for (a,b),checked in zip(regions,continuity['regions']):
                    if checked['mix']==0:
                        continue
                    lo, hi = max(a, start), min(b, start + n)
                    if lo >= hi:
                        continue
                    pos = np.arange(lo, hi)
                    w = fade(pos,a,b)*checked['mix']
                    s = slice(lo - start, hi - start)
                    y[s] += w * (x[s] - y[s])
                if not np.isfinite(y).all() or np.max(abs(y), initial=0) >= 1:
                    raise ValueError('Protected output would clip')
                out.write(y)
        if hashes != [_hash(p) for p in (source, base, gentle, review_receipt)]:
            raise ValueError('An input changed during rendering')
        if sf.info(scratch).frames != infos[0].frames:
            raise ValueError('Output duration changed')
        metadata = {'recipe': 'folio-gemini-phrases-v2-guarded',
                    'source_sha256': hashes[0], 'base_sha256': hashes[1], 'gentle_sha256': hashes[2],
                    'review_sha256': hashes[3], 'output_sha256': _hash(scratch),
                    'samples': infos[0].frames, 'plan': plan,
                    'regions_seconds': [[a / 48000, b / 48000] for a, b in regions],
                    'normal_narration_samples_outside_regions_unchanged': True,
                    'baseline_pitch_hz': decision['baseline_pitch_hz'],
                    'pause_padding_ms': 140, 'fade_ms': 120, 'alignment_samples': delay,
                    'gentle_matching_gain_db': gain_db, 'gentle_eq_profile': profile,
                    'continuity_control': continuity,
                    'implementation_sha256': _hash(Path(__file__)),
                    'continuity_implementation_sha256': _hash(Path(__file__).with_name('continuity.py')),
                    'status': 'Experimental Gemini-directed local rendering. No claim of Auphonic equivalence or human approval.'}
        # Exclusive publication, preserving all original files.
        with scratch.open('rb') as inp, target.open('xb') as out:
            import shutil
            shutil.copyfileobj(inp, out, 1048576)
        with target.with_suffix('.json').open('x') as f:
            f.write(json.dumps(metadata, indent=2, allow_nan=False) + '\n')
    return metadata

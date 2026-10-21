"""Contextual overlap-add for model outputs that preserve the input clock."""

import numpy as np


def restore_file(source, target, infer, core_seconds=3., context_seconds=.5,
                 crossfade_seconds=.08, progress=lambda seconds: None):
    """File-backed equivalent of restore_chunks, retaining its global grid.

    Keeps at most one core plus two context margins in memory. The output is
    unnormalized DOUBLE PCM; callers can measure the whole-file peak and apply
    one safety gain on export. No second layer of independently timed chunks.
    """
    import soundfile as sf
    from pathlib import Path
    if Path(source).resolve() == Path(target).resolve():
        raise ValueError('Output must differ from source')
    if Path(target).exists():
        raise FileExistsError(target)
    rate = sf.info(source).samplerate
    stride, context, fade = [round(t*rate) for t in
                             (core_seconds,context_seconds,crossfade_seconds)]
    if stride <= 2*fade or context < fade:
        raise ValueError('Invalid chunk context or crossfade')
    pending = np.empty(0)
    pending_weight = np.empty(0)
    count = 0
    peak = 0.
    with sf.SoundFile(source) as inp, sf.SoundFile(
            target,'w',samplerate=rate,channels=1,subtype='DOUBLE',format='RF64') as out:
        if inp.channels != 1:
            raise ValueError('Expected mono source')
        total = len(inp)
        for start in range(0,total,stride):
            end = min(total,start+stride)
            a,b = max(0,start-context),min(total,end+context)
            inp.seek(a)
            block = inp.read(b-a,dtype='float64')
            generated = np.asarray(infer(block),dtype='float64').reshape(-1)
            left,right = max(0,start-fade),min(total,end+fade)
            segment = generated[left-a:right-a]
            if len(segment) != right-left or not np.isfinite(segment).all():
                raise ValueError('Invalid model chunk or missing context')
            positions = np.arange(left,right)
            weight = np.ones(right-left)
            if start > 0 and fade:
                weight = np.minimum(weight,np.clip((positions-(start-fade))/(2*fade),0,1))
            if end < total and fade:
                weight = np.minimum(weight,np.clip(((end+fade)-positions)/(2*fade),0,1))
            values = segment*weight
            values[:len(pending)] += pending
            weight[:len(pending_weight)] += pending_weight
            stop = total if end == total else end-fade
            emit = stop-left
            if np.any(weight[:emit] <= 0):
                raise ValueError('Uncovered samples in file-backed restoration')
            y = values[:emit]/weight[:emit]
            out.write(y)
            count += len(y)
            peak = max(peak,float(np.max(abs(y),initial=0)))
            pending = values[emit:].copy()
            pending_weight = weight[emit:].copy()
            progress(end/rate)
        if count != total:
            raise ValueError('File-backed restoration lost samples')
    return {'samples':count,'sample_rate':rate,'peak':peak,
            'maximum_input_window_seconds':(stride+2*context)/rate}


def restore_chunks(
    audio,
    rate,
    infer,
    core_seconds=3.0,
    context_seconds=0.5,
    crossfade_seconds=0.04,
    output_factor=1,
):
    stride = round(core_seconds * rate)
    context = round(context_seconds * rate)
    fade = round(crossfade_seconds * rate)
    if stride <= 2 * fade or context < fade:
        raise ValueError("Invalid chunk context or crossfade")
    n = len(audio) * output_factor
    out = np.zeros(n)
    weight = np.zeros(n)
    for start in range(0, len(audio), stride):
        end = min(len(audio), start + stride)
        a = max(0, start - context)
        b = min(len(audio), end + context)
        generated = np.asarray(infer(audio[a:b]), dtype="float64").reshape(-1)
        left = max(0, start - fade)
        right = min(len(audio), end + fade)
        offset = (left - a) * output_factor
        count = (right - left) * output_factor
        segment = generated[offset : offset + count]
        if len(segment) != count or not np.isfinite(segment).all():
            raise ValueError("Invalid model chunk or missing context")
        positions = (
            np.arange(left * output_factor, right * output_factor) / output_factor
        )
        w = np.ones(count)
        if start > 0 and fade:
            w = np.minimum(w, np.clip((positions - (start - fade)) / (2 * fade), 0, 1))
        if end < len(audio) and fade:
            w = np.minimum(w, np.clip(((end + fade) - positions) / (2 * fade), 0, 1))
        out[left * output_factor : right * output_factor] += segment * w
        weight[left * output_factor : right * output_factor] += w
    if np.any(weight <= 0):
        raise ValueError("Uncovered samples in chunked restoration")
    return out / weight

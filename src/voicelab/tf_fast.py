"""Faster forward passes for the pinned TF-Restormer checkpoint.

The vendored model source is verified against a pinned revision and is never
edited. These helpers change how the same weights are evaluated, from inside
Folio's own inference process:

* ``lean_attention`` computes the same scaled dot-product attention without the
  masking bookkeeping that PyTorch's generic CPU path performs for every call.
* ``slice_blocks`` evaluates each time/frequency block on a few sequences at a
  time. Sequences never interact inside a block, so the result is the same;
  the intermediates stay near cache size instead of spanning gigabytes.

Both keep the published weights and the model's arithmetic. Results can differ
from the stock forward pass by float32 rounding only.
"""
from __future__ import annotations

import math

import torch
from torch import nn


def _attention(query, key, value):
    """softmax(q k^T / sqrt(d)) v, with the scale split as PyTorch does."""
    scale = math.sqrt(1/math.sqrt(query.size(-1)))
    weights = torch.matmul(query*scale, (key*scale).transpose(-2, -1))
    weights = torch.softmax(weights, dim=-1)
    return torch.matmul(weights, value)


def _self_attention_forward(self, input):
    query, key, value = self.get_qkv(input)
    if hasattr(self, 'rope'):
        query, key = self.apply_rope(query, key)
    output = _attention(query, key, value).transpose(1, 2)
    return self.aggregate_heads(output.reshape(output.shape[:2]+(-1,)))


def _linformer_forward(self, input, kv=None):
    if not self.selfattn and kv is None:
        raise ValueError('Cross-attention needs key/value input')
    query, key, value = self.get_qkv(input, kv)
    if hasattr(self, 'rope'):
        query, key = self.apply_rope(query, key)
    key, value = self.Ekv(key, value)
    output = _attention(query, key, value).transpose(1, 2)
    return self.aggregate_heads(output.reshape(output.shape[:2]+(-1,)))


def lean_attention(model):
    """Route every attention layer of ``model`` through ``_attention``."""
    import types
    count = 0
    for module in model.modules():
        name = type(module).__name__
        if name == 'MultiHeadSelfAttention':
            module.forward = types.MethodType(_self_attention_forward, module)
        elif name == 'LinformerAttention':
            module.forward = types.MethodType(_linformer_forward, module)
        else:
            continue
        if getattr(module, 'dropout', 0.):
            raise ValueError('Lean attention expects an inference checkpoint without dropout')
        count += 1
    return count


class Sliced(nn.Module):
    """Evaluate a per-sequence block on ``size`` sequences at a time."""

    def __init__(self, block, size):
        super().__init__()
        self.block = block
        self.size = int(size)

    def forward(self, x, kv=None):
        count = x.shape[0]
        if count <= self.size:
            return self.block(x) if kv is None else self.block(x, kv)
        output = None
        for start in range(0, count, self.size):
            stop = min(count, start+self.size)
            part = (self.block(x[start:stop]) if kv is None
                    else self.block(x[start:stop], kv[start:stop]))
            if output is None:
                output = part.new_empty((count,)+part.shape[1:])
            output[start:stop] = part
        return output


def slice_blocks(model, frequency=8, time=64):
    """Wrap the frame-wise and bin-wise blocks so they run in small slices.

    ``frequency`` is the number of frames handled together by a block that
    attends across frequency bins; ``time`` is the number of bins handled
    together by a block that attends across frames.
    """
    count = 0
    for stage in (model.encoder, model.decoder):
        for block in stage.tf_block:
            for holder, size in ((block.frame_wise_block, frequency),
                                 (block.freq_wise_block, time)):
                if size and not isinstance(holder.block, Sliced):
                    holder.block = Sliced(holder.block, size)
                    count += 1
    return count


def slice_sizes(device):
    """(frames per frequency block, bins per time block) for a device type.

    CPU slices are sized to stay near the cache; a GPU only needs the peak
    memory bounded, so its slices are much larger.
    """
    return (8, 64) if str(device) == 'cpu' else (64, 512)


RESAMPLE = {48000: (1, 1), 32000: (2, 3), 24000: (1, 2), 16000: (1, 3), 8000: (1, 6)}


def restore_window(inference, block, rate=48000, precision='float32'):
    """One window through the model at ``rate`` input bandwidth, 48 kHz out.

    Mirrors the vendored single-pass path (STFT, network, inverse STFT). The
    transforms always run in float32; ``precision='float16'`` autocasts only
    the network. Returns a float64 array with exactly ``len(block)`` samples.
    """
    import numpy as np
    from scipy import signal
    engine = inference.engine
    samples = np.asarray(block, dtype='float64')
    up, down = RESAMPLE[rate]
    if rate != 48000:
        samples = signal.resample_poly(samples, up, down)
    x = torch.as_tensor(samples, dtype=torch.float32, device=engine.device)[None, :]
    spectrum = engine.stft[str(rate)](x, cplx=True)
    stacked = torch.stack([spectrum.real, spectrum.imag], dim=-1)
    if precision == 'float16':
        with torch.autocast(engine.device.type, dtype=torch.float16):
            restored = engine.model(stacked, out_F=engine.out_F)
    else:
        restored = engine.model(stacked, out_F=engine.out_F)
    restored = restored.float()
    wave = engine.istft['48000'](torch.complex(restored[..., 0], restored[..., 1]),
                                 cplx=True, squeeze=False)
    y = wave.detach().cpu().numpy().reshape(-1).astype('float64')
    if len(y) < len(block):
        y = np.pad(y, (0, len(block)-len(y)))
    return y[:len(block)]

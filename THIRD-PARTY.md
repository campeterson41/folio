# Third-party components

## Source pipeline (`bin/folio-source`)

The CPU source pipeline contains no third-party code, model, weights or audio.
It depends at runtime only on [NumPy](https://numpy.org/) (BSD-3-Clause),
[SciPy](https://scipy.org/) (BSD-3-Clause) and
[python-soundfile](https://github.com/bastibe/python-soundfile) (BSD-3-Clause;
it uses libsndfile, LGPL-2.1-or-later), installed separately from PyPI.

Its methods are textbook signal processing implemented in Folio's own code:
STFT spectral subtraction with a smoothed, floored gain; windowed-FIR
equalization (`scipy.signal.firwin2`); and a delayed-power late-tail
attenuation loosely motivated by
[Lebart, Boucher & Denbigh (2001)](https://www.ee.columbia.edu/~dpwe/papers/LebBD01-ssdereverv.pdf).
Folio does not implement that paper's room-decay model or estimation, and
reuses none of its code. The numerical tone/gain preset is a fixed local setting.

## Neural restoration

Folio's optional neural path runs the independently published TF-Restormer
model. No model weights are redistributed in this repository; the installer downloads the pinned public
artifacts and verifies their SHA-256 hashes before inference.

| Component | Source | License evidence |
| --- | --- | --- |
| TF-Restormer implementation | [dmlguq456/TF_Restormer](https://github.com/dmlguq456/TF_Restormer), commit `eaefe1cb3811c13f324eee34a81564cff8b7da7e` | MIT; upstream notice retained at [licenses/TF_Restormer.txt](licenses/TF_Restormer.txt) |
| Selected checkpoint | [shinuh/tf-restormer-baseline](https://huggingface.co/shinuh/tf-restormer-baseline), revision `bd8ffc3105299ea5497797783ed1ce750e089523`, corrected epoch 19 export | Upstream model card declares MIT |
| Optional DeepFilterNet denoiser | [Rikorose/DeepFilterNet](https://github.com/Rikorose/DeepFilterNet), pinned v0.5.6 Linux executable; SHA-256 in `resources/conservative-lock.json` | Upstream MIT notice retained at [licenses/DeepFilterNet.txt](licenses/DeepFilterNet.txt); binary downloaded separately |

PyTorch, NumPy, SciPy, SoundFile, uv, Python and other dependencies retain their
own licenses. NVIDIA runtime wheels have separate NVIDIA terms. Folio's MIT
license does not relicense those components. The Linux CUDA dependency lock is
an installation recipe, not a redistribution of those binaries.

The optional low-shelf EQ uses the standard RBJ biquad equations. There are no
Auphonic weights, service code, voice recordings, credentials or service outputs
in the repository. The numerical preset is a fixed local EQ/dynamics setting.

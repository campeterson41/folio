# Third-party components

Folio runs the independently published TF-Restormer model. No model weights are
redistributed in this repository; the installer downloads the pinned public
artifacts and verifies their SHA-256 hashes before inference.

| Component | Source | License evidence |
| --- | --- | --- |
| TF-Restormer implementation | [dmlguq456/TF_Restormer](https://github.com/dmlguq456/TF_Restormer), commit `eaefe1cb3811c13f324eee34a81564cff8b7da7e` | MIT; upstream notice retained at [licenses/TF_Restormer.txt](licenses/TF_Restormer.txt) |
| Selected checkpoint | [shinuh/tf-restormer-baseline](https://huggingface.co/shinuh/tf-restormer-baseline), revision `bd8ffc3105299ea5497797783ed1ce750e089523`, corrected epoch 19 export | Upstream model card declares MIT |

PyTorch, NumPy, SciPy, SoundFile, uv, Python and other dependencies retain their
own licenses. NVIDIA runtime wheels have separate NVIDIA terms. Folio's MIT
license does not relicense those components. The Linux CUDA dependency lock is
an installation recipe, not a redistribution of those binaries.

The optional low-shelf EQ uses the standard RBJ biquad equations. There are no
Auphonic weights, service code, voice recordings, credentials or service outputs
in the repository. The numerical preset is a fixed local EQ/dynamics setting.

# Folio

Experimental, local voice restoration for audiobook narration.

Folio combines the open-source **TF-Restormer** model with EQ, level control and
automatic preservation of consonants that restoration may suppress. Audio stays
on your computer. No subscription, API key or paid service is required.

This is an early **Python command-line processor**, separate from a recording
or editing application. It is not an Auphonic implementation or affiliated with
Auphonic. Comparable sound quality has not been established.

## Quick start

The tested installer supports **Linux x86_64** and Python 3.11.17. A compatible
NVIDIA driver is recommended: the pinned inference wheels use CUDA 12.1.
Git and a system Python 3.11 or later are needed to run setup. Setup downloads
several GB of public dependencies and model artifacts into `~/.local/share/folio`.
It does not access recordings or accounts.

```sh
git clone https://github.com/campeterson41/folio.git
cd folio
python tools/setup_f_runtime.py
bin/folio --input narration.wav --output restored.wav --preserve-voiced-fricatives --finish-levels
```

Input currently must be **48 kHz mono WAV**. Output is PCM-24 WAV, with the same
sample count and a JSON processing receipt. Originals and existing output files
are preserved.

For a small bass lift:

```sh
bin/folio --input narration.wav --output warmer.wav \
  --preserve-voiced-fricatives --bass-db 0.5 --finish-levels
```

`--bass-db 0.5` adds a gentle 160 Hz low shelf. The default is no extra bass.
`--finish-levels` adds optional final RMS/peak control. Its -20 dBFS RMS target
can remain unmet when the -4 dBFS peak ceiling takes precedence.
A loud near-microphone test section can hold the rest of a take down when final
level control is disabled. The commands above enable it; it does not cut that
section from the recording.
`--device cpu` is available; chapter-length CPU performance has not been validated.

## What happens to the sound?

1. The pinned TF-Restormer checkpoint processes overlapping windows with context.
2. A saved numerical EQ/dynamics preset and subtle presence adjustment are applied.
3. A signal-driven detector can preserve tone-matched source speech where the model
   appears to have suppressed fricative energy. It uses no words or repair timestamps.
4. Optional bass EQ and final level control follow.

The base detector is conservative. `--preserve-voiced-fricatives` enables the
experimental Automatic 2 detector, including quieter voiced fricatives. The
supplied tone preset was tuned for one speaker and microphone; it is not
personalized automatically for every voice.

## Current limits

- Neural restoration can alter consonants or vocal texture. Preservation remains
  experimental and does not guarantee faithful pronunciation.
- Early listening tests found muddling on acted and higher-register voices in the
  current checkpoint. EQ alone cannot guarantee a repair; alternatives remain under evaluation.
- Short bursts of high-frequency interference may be mistaken for lost speech and
  restored from the source. Steady-noise tests do not cover this failure mode.
- There is no recording UI, waveform editor, automatic chapter splitting or audiobook
  publishing certification in this release.
- One independent 10-minute-52-second take has been processed with consistent sample
  timing. Audio windows are bounded; global 10 ms decision arrays grow with duration.
- Passing tests, similar spectra and speech-recognition agreement are not listening proof.

Keep an original, and audition the result before using it in a book.

## Reproducibility

```sh
python tools/setup_f_runtime.py --verify-only
```

`resources/f-model-lock.json` pins the source commit and model/config hashes.
`requirements-f.lock` freezes 31 inference dependencies. Setup preserves unknown
existing environments or mismatching artifacts rather than replacing them.
`VOICE_LAB_RUNTIME=/path/to/runtime` selects another installation location.
Setup `--offline` uses existing caches and fails if an artifact is missing.

## Tests

The tests generate synthetic signals; no personal recordings or model weights
are included in this repository. Most tests run without a GPU or model download.

```sh
python -m venv .venv-test
.venv-test/bin/python -m pip install -r requirements-test.txt
.venv-test/bin/python -m pytest -q
```

## License and credits

Folio's original code is MIT-licensed. TF-Restormer and the selected public
checkpoint are independently published upstream; their credits and pinned
sources are in [THIRD-PARTY.md](THIRD-PARTY.md).

Contributions and reports of general failure cases are welcome. Please don't
post private recordings or recordings you lack permission to share in issues.

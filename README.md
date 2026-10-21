# Folio

Experimental, local voice restoration for audiobook narration.

Folio combines the open-source **TF-Restormer** model with EQ, level control and
automatic preservation of consonants that restoration may suppress. Audio stays
on your computer by default. No subscription, API key or paid service is required
for restoration. An optional, explicitly invoked Gemini reviewer can receive
short excerpts, as described below.

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

## Experimental adaptive mode

Adaptive mode analyzes 40 ms windows every 20 ms, estimates the recording's usual
vocal register and chooses a gentler, source-preserving denoiser for higher-register
phrases. It learns a restrained EQ match from intact narration in this recording,
smooths transitions, aligns the two paths and applies final level control.

```sh
python tools/setup_f_runtime.py --with-conservative
bin/folio --input narration.wav --output adaptive.wav \
  --preserve-voiced-fricatives --bass-db 0.5 --adaptive
```

The additional pinned DeepFilterNet executable is about 36 MB. Its model receives
bounded windows with context. The output receipt includes routing regions, a
half-second analysis timeline, learned EQ and gain decisions. Processing uses
neither transcripts nor manually supplied word locations.

This controller is a local heuristic, **not an AI listening verdict**. Pitch is
a routing hint; it does not prove that speech is damaged. Adaptive mode remains
opt-in and needs listening validation across voices and microphones. It never
calls a cloud provider itself.

Listening feedback found abrupt/choppy changes in this first adaptive mode.
It remains experimental; a passing waveform or playback test does not establish
transparent restoration transitions.

## Optional Gemini listening review

`bin/folio-review` prepares matching raw/current/adaptive excerpts and blinds the
two processed versions as A and B. It matches levels with constant gain, leaving
EQ and dynamics untouched. The default is a local preparation report: **nothing
is sent unless you add `--send`**.

```sh
bin/folio-review --raw narration.wav --current restored.wav \
  --adaptive adaptive.wav --receipt adaptive.json --output review-plan.json
```

The adaptive receipt selects up to three diverse 12-second excerpts: higher
register, ordinary narration and a loud event when detected. Inspect their start times in the
plan, or choose exact excerpts with repeated `--excerpt START:DURATION` arguments
instead of `--receipt`. The hard limit is three excerpts of 15 seconds each,
across three versions (135 seconds of audio total). No complete recording is sent.

For an actual review, provide `GEMINI_API_KEY` through your own scoped credential
loader or child-process environment, repeat the command with `--send` and a new
output filename. Folio does not store the key. The default model is
`gemini-3.5-flash-lite`, with a single request and at most 1,800 output tokens.
Upload attempts are recorded before the network call, including failed/rejected
results; there are no automatic retries. See Google's current
[pricing](https://ai.google.dev/gemini-api/docs/pricing) and
[data terms](https://ai.google.dev/gemini-api/terms) for your account tier.

The review contains observations and restrained EQ/restoration suggestions.
Suggestions are validated (EQ within ±1 dB) and **never applied automatically**.
The receipt's `advice_applies_to` explicitly names which processed version was B.
It cannot issue processing commands or patch words. Model judgments can miss
small artifacts or be wrong; this is an experimental second opinion, not a
substitute for listening. Network request handling is covered by synthetic tests;
live provider access requires your own valid key.

## Experimental Gemini-directed rendering

An actual completed audio review can drive a separate local render:

```sh
bin/folio-ai --input narration.wav --base restored.wav \
  --gentle-stem conservative.wav --review review.json --output ai-guided.wav
```

This currently requires a verified DeepFilterNet conservative stem/receipt and
a review whose blinded **B** candidate is the exact supplied Folio base. It
rejects missing, failed, unrelated or wholly uncertain review advice. The base
must be PCM-16/24 so samples outside protected regions remain exact.

If Gemini recommends gentler restoration on higher-register examples, the
renderer infers complete speech regions from the original recording. It uses
the gentler path throughout each selected region, with transitions padded into
pauses, a robust fixed path-matching gain, and restrained EQ advice. It does not
convert the review excerpts into timestamp-specific patches. A near-mic overload
keeps its complete utterance on the existing peak-safe path.

Gemini supplies advice; local models and DSP produce the audio. It does not
generate a new performance or recover Auphonic's proprietary model. Automated
listening opinions can be vague or wrong. This mode is experimental and needs
human audition, especially across character voices and changing microphone distance.

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

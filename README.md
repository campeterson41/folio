# Folio

Experimental, local narration cleanup for audiobook recordings.

The recommended path is **`folio-source`**: a CPU-only processor that works on
your recording itself. It uses no neural model, no resynthesis, no reference
recording, no transcript and no network. It needs only NumPy, SciPy and
SoundFile. Older neural and Gemini-assisted modes remain available below as
optional experiments.

Folio is a Python command-line processor, separate from a recording or editing
application. It is not an Auphonic implementation or affiliated with Auphonic,
and comparable sound quality has not been established.

## Quick start: source pipeline (CPU)

Python 3.11 is tested. From a clone:

```sh
git clone https://github.com/campeterson41/folio.git
cd folio
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements-source.txt
bin/folio-source --input narration.wav --output cleaned.wav
```

`bin/folio-source` uses `FOLIO_PYTHON` if set, otherwise `.venv/bin/python`
when present, otherwise `python3`.

- Input must be a **48 kHz mono WAV**. Convert other formats first; Folio does
  not resample or mix down for this path.
- The take needs **quiet room tone**: several unbroken pauses of at least 0.3 s
  that contain only background noise. Noise is learned from the quietest quarter
  of the recording. Very short takes, or continuous noise/music with no pauses,
  are refused with an explanation rather than processed blindly.
- Output is a 24-bit PCM file in the RF64 (64-bit WAV) container, with exactly
  the input's sample count, plus a JSON receipt beside it (`cleaned.json`).
  Most editors and FFmpeg read RF64; for a tool that needs a classic WAV header,
  `ffmpeg -i cleaned.wav -c:a pcm_s24le cleaned-classic.wav` converts it
  losslessly. Existing files are never overwritten, the
  input is hash-checked before and after, and nothing is published if processing
  fails or is interrupted.

On a 12-thread development laptop, a five-minute synthetic take took about 80
seconds; time grows roughly in proportion to length.

### What the source pipeline does

All stages apply gains or fixed filters to the original performance; nothing is
generated, replaced, re-timed or edited.

1. **Noise estimate.** Median spectrum of up to 256 quiet, non-pitched moments.
2. **Analysis branch.** A gentle (≤12 dB) spectral noise reduction, 50 Hz
   high-pass, the preset's broad tone curve, its fixed gain and a −4 dBFS peak
   limiter. This branch is only measured to steer level control.
3. **Quieter branch.** The same chain from the raw input with stronger noise
   subtraction (1.6×, ≤18 dB). Gains are positive and real, so the source
   phase in each STFT bin is kept.
4. **Level control.** Gentle 1.5:1 compression of loud speech (at most 2.5 dB) and up
   to 9 dB of soft reduction in pauses, with a guard so detected syllables and
   consonants are not reduced.
5. **Closer, warmer finish.** Up to 6 dB of attenuation where spectral energy is
   falling after louder sound (a late room-tail heuristic), then a 3 dB low shelf
   centred at 160 Hz.

Processing streams aligned 30-second blocks with 2 seconds of context, so long
chapters use bounded audio memory. With the pinned library versions, the same
input and settings reproduce the same output file.

### Settings

```sh
bin/folio-source --input narration.wav --output cleaned.wav --tail-db 3 --bass-db 1.5
```

| Setting | Default | Effect |
| --- | --- | --- |
| `--tail-db` | 6 (0–6) | Less lingering room sound. Higher values can also soften quiet word endings and breaths; 0 disables it. |
| `--bass-db` | 3 (0–4) | Warmth below roughly 160 Hz. Lower it for boomy rooms or proximity effect. |
| `--preset` | `presets/source-narration-v1.json` | Broad tone curve (±6 dB) and fixed gain. |

The supplied preset was **developed for one speaker and one microphone**,
and the defaults were preferred on one real recording. That is a single
listening approval, not a benchmark. Other voices, rooms and microphones may
need a different tone curve, gain, tail or bass setting. To adjust the tone,
copy the preset and edit `tone.gains_db` (dB at the listed `frequencies_hz`)
or `gain_db`; there is no automatic loudness normalization in this path.

### Limits of the source pipeline

- The room-tail reduction is a **fixed heuristic**: it compares each band with a
  delayed, decaying average of its own recent power. It does not estimate the
  room's impulse response or separate reverberation exactly, and it can reduce
  quiet speech that follows louder sound. It is loosely motivated by spectral
  subtraction for late reverberation ([Lebart, Boucher & Denbigh,
  2001](https://www.ee.columbia.edu/~dpwe/papers/LebBD01-ssdereverv.pdf)) but
  does not implement that paper's estimator.
- Noise reduction assumes steady background noise learned once per take. It can
  leave a mild "watery" texture on hiss-heavy takes and does not remove sudden
  noises, mouth clicks or noise that changes during the recording.
- Very loud passages are held near −4 dBFS by a peak limiter before the bass
  shelf; the receipt warns when it reduces peaks by more than 6 dB. Output whose
  estimated true peak would reach −0.1 dBFS is refused rather than clipped.
  Clipping already in the recording cannot be repaired.
- No retake removal, speech editing or chapter splitting is performed.
- Passing tests are synthetic-signal checks, not listening proof. Keep the
  original and audition the result before using it in a book.

## Optional: neural restoration (TF-Restormer, GPU)

The earlier, heavier path combines the open-source **TF-Restormer** model with
EQ, level control and automatic preservation of consonants that restoration
may suppress. Audio stays on your computer by default. An optional, explicitly
invoked Gemini reviewer can receive short excerpts, as described below.

### Setup

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

### What happens to the sound?

1. The pinned TF-Restormer checkpoint processes overlapping windows with context.
2. A saved numerical EQ/dynamics preset and subtle presence adjustment are applied.
3. A signal-driven detector can preserve tone-matched source speech where the model
   appears to have suppressed fricative energy. It uses no words or repair timestamps.
4. Optional bass EQ and final level control follow.

The base detector is conservative. `--preserve-voiced-fricatives` enables the
experimental Automatic 2 detector, including quieter voiced fricatives. The
supplied tone preset was tuned for one speaker and microphone; it is not
personalized automatically for every voice.

### Experimental adaptive mode

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

### Optional Gemini listening review

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

### Experimental Gemini-directed rendering

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
one fixed blend of the gentler path throughout each selected region, with transitions padded into
pauses, a robust fixed path-matching gain, and restrained EQ advice. It does not
convert the review excerpts into timestamp-specific patches. A near-mic overload
keeps its complete utterance on the existing peak-safe path.

The first full-replacement experiment introduced dropouts from gaps in the
restoration stem. The renderer now measures each proposed blend against the
supplied base before export: 20 ms full-band and 300–3000 Hz energy windows,
sampled every 5 ms, must stay within a 1 dB loss and 3 dB boost wherever the base
exceeds the recording-derived audible floor. The check includes crossfade phase
cancellation. An unsafe proposal reduces the blend for the entire utterance or
keeps the base unchanged; it never switches paths on each failing frame.
The receipt records proposed and applied changes and the selected blend.
These measured limits protect continuity relative to the base; they do not prove
better EQ, faithful consonants or perceptual quality.

Gemini supplies advice; local models and DSP produce the audio. It does not
generate a new performance or recover Auphonic's proprietary model. Automated
listening opinions can be vague or wrong. This mode is experimental and needs
human audition, especially across character voices and changing microphone distance.

### Limits of the neural path

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

### Neural runtime reproducibility

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
are included in this repository. Most tests run without a GPU or model download;
the source-pipeline tests need only `requirements-test.txt`.

```sh
python -m venv .venv-test
.venv-test/bin/python -m pip install -r requirements-test.txt
.venv-test/bin/python -m pytest -q
```

## License and credits

Folio's original code is MIT-licensed. The source pipeline uses only NumPy,
SciPy and SoundFile. TF-Restormer and the selected public checkpoint are independently published upstream; their credits and pinned
sources are in [THIRD-PARTY.md](THIRD-PARTY.md).

Contributions and reports of general failure cases are welcome. Please don't
post private recordings or recordings you lack permission to share in issues.

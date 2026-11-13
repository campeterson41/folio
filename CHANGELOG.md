# Changelog

## Unreleased — faster long takes, speed/quality tiers

- Process independent audio windows in a worker pool, keep the numeric
  libraries single-threaded for the tiny per-bin solves, measure true peak only
  where it is read, and run the model beside the source-only stages. The
  `reference` render no longer depends on worker or core count; against 0.3.0 it
  was byte-identical on the 60 s and ten-minute takes tested on Linux and within
  rounding (-97 dB) on a five-minute take on macOS, because 0.3.0's result
  varied with the BLAS thread count.
  `bin/folio-source` uses the same pool and its output file is unchanged.
- Delete intermediate stems as soon as no later stage reads them, and stop
  syncing each one to disk. Peak temporary disk drops from about 1.1 GiB to
  about 0.35 GiB per input minute.
- Add `--quality reference|high|balanced|fast|draft`, `--workers` and `--threads`.
  `reference` remains the default and the approved sound. Receipts record the
  tier, model input rate, precision, threads and per-window model time.
- CPU inference no longer runs attention-mask bookkeeping (bit-identical), and
  tiers other than `reference` evaluate blocks in cache-sized slices, cutting
  peak model memory from about 6 GiB to about 1.2 GiB.
- Add `tools/benchmark_studio.py` to time tiers on a take and measure each
  render's distance from the reference render.

## 0.3.0 — Folio Studio

- Make `bin/folio` run the official Studio recipe, also available as
  `bin/folio-studio`. Earlier restoration remains at `bin/voice-lab-f`;
  `bin/folio-source` keeps its existing CPU-only behavior.
- Freeze the 22% restored-voice blend, parallel vocal compression, room
  prediction, speech-protected final cleanup and very quiet synthetic floor.
- Add an isolated, pinned Studio installer with cache-only and verification
  modes. Processing itself needs no network, reference audio or credentials.
- Support optional local edit/repair plans and verified native-inference reuse.
  New recordings use the current automatic repair detector; the prototype's
  earlier repair decisions are not redistributed or claimed to regenerate.
- Add synthetic checks for streaming dynamics, edit clocks, cleanup bounds,
  deterministic noise, cached-source identity and atomic output publication.

This changes the default command's sound and flags. Existing scripts using
`--adaptive`, `--finish-levels`, `--bass-db` or
`--preserve-voiced-fricatives` should call `bin/voice-lab-f` explicitly.

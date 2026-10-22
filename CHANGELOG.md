# Changelog

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

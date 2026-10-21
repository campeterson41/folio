#!/usr/bin/env python3
"""Prepare or explicitly send short raw/current/adaptive excerpts to Gemini."""
import argparse
import json
import os
from pathlib import Path
import soundfile as sf
from voicelab.listening_review import MODEL, review, select_windows, _hash


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--raw', type=Path, required=True)
    ap.add_argument('--current', type=Path, required=True)
    ap.add_argument('--adaptive', type=Path, required=True)
    ap.add_argument('--receipt', type=Path, help='Adaptive receipt for automatic excerpt selection')
    ap.add_argument('--excerpt', action='append', help='START:DURATION seconds; at most three, each <=15 seconds')
    ap.add_argument('--model', default=MODEL)
    ap.add_argument('--send', action='store_true', help='Explicitly send selected audio to Google using GEMINI_API_KEY')
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    protected = [args.raw, args.current, args.adaptive] + ([args.receipt] if args.receipt else [])
    if args.output.exists() or args.output.resolve() in [p.resolve() for p in protected]:
        ap.error('Output exists or aliases an input')
    if args.excerpt:
        if args.receipt:
            ap.error('Choose --excerpt or --receipt')
        try:
            windows = [tuple(map(float, x.split(':'))) for x in args.excerpt]
        except ValueError:
            ap.error('Excerpt format is numeric START:DURATION')
        if any(len(w) != 2 for w in windows):
            ap.error('Excerpt format is START:DURATION')
    elif args.receipt:
        receipt = json.loads(args.receipt.read_text())
        if (receipt.get('source_sha256') != _hash(args.raw) or
                receipt.get('output_sha256') != _hash(args.adaptive)):
            ap.error('Adaptive receipt does not match the supplied recordings')
        windows = select_windows(receipt, sf.info(args.raw).duration)
    else:
        ap.error('Supply --excerpt or --receipt')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as f:
        def checkpoint(result):
            f.seek(0)
            json.dump(result, f, indent=2, allow_nan=False)
            f.write('\n')
            f.truncate()
            f.flush()
            os.fsync(f.fileno())
        checkpoint({'status': 'not-sent-preparing', 'recommendations_applied': False})
        try:
            result = review([args.raw, args.current, args.adaptive], windows,
                            args.model, args.send, recorder=checkpoint)
        except (ValueError, OSError) as exc:
            checkpoint({'status': 'not-sent-preflight-failed', 'error_type': type(exc).__name__})
            ap.exit(2, 'Review preparation failed; no audio sent. Check inputs and scoped credentials.\n')
        checkpoint(result)
    print(result['status'], args.output)
    if result['status'] not in ('reviewed', 'prepared-not-sent'):
        ap.exit(2, 'Review attempt recorded; no recommendations accepted. No automatic retry.\n')


if __name__ == '__main__':
    main()

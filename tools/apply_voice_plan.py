#!/usr/bin/env python3
"""Render a completed Gemini review as a bounded, local voice-protection policy."""
import argparse
from pathlib import Path
from voicelab.voice_plan import apply
from voicelab.conservative import validate
from voicelab.listening_review import _hash
import soundfile as sf


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input', type=Path, required=True)
    ap.add_argument('--base', type=Path, required=True)
    ap.add_argument('--gentle-stem', type=Path, required=True)
    ap.add_argument('--review', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    validate(args.gentle_stem, _hash(args.input), sf.info(args.input).frames)
    result = apply(args.input, args.base, args.gentle_stem, args.review, args.output)
    print(result['recipe'], args.output)


if __name__ == '__main__':
    main()

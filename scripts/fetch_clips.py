#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Fetch a small test set from Google FLEURS (CC-BY-4.0) without downloading whole splits.

FLEURS ships each language's test audio as one tar.gz on Hugging Face. This streams the archive, keeps the first
utterances it meets, and stops, so a language costs a few MB instead of hundreds. For each language it writes:

  <out>/<lang>_fleurs_<n>.wav     short utterances, 16 kHz mono (n = 1..--short)
  <out>/<lang>_fleurs_long.wav    --long-seconds of further utterances joined with 0.5 s of silence, to exercise
                                  windowing
  <out>/manifest.json             [{id, lang, path, seconds, reference}] with FLEURS' normalised transcription

Your own recordings can be added to the manifest by hand (reference may be empty).

Usage: python fetch_clips.py --out clips --langs te ta hi --short 3 --long-seconds 90
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import soundfile as sf

FLEURS = "https://huggingface.co/datasets/google/fleurs/resolve/main/data/{cfg}/{what}"
CONFIGS = {"te": "te_in", "ta": "ta_in", "hi": "hi_in", "kn": "kn_in", "ml": "ml_in", "mr": "mr_in",
           "bn": "bn_in", "gu": "gu_in", "pa": "pa_in", "or": "or_in", "ur": "ur_pk", "as": "as_in"}
RATE = 16000


def read_audio(data: bytes) -> np.ndarray:
    audio, rate = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if rate != RATE:
        import librosa  # only needed if FLEURS ever changes rate
        audio = librosa.resample(audio, orig_sr=rate, target_sr=RATE)
    return audio.astype(np.float32)


def fetch(lang: str, out: Path, short: int, long_seconds: float) -> list[dict]:
    cfg = CONFIGS[lang]
    tsv = urllib.request.urlopen(FLEURS.format(cfg=cfg, what="test.tsv")).read().decode()
    refs = {row[1]: row[3] for row in csv.reader(io.StringIO(tsv), delimiter="\t", quoting=csv.QUOTE_NONE)}
    entries, long_parts, long_refs, long_len = [], [], [], 0.0
    with urllib.request.urlopen(FLEURS.format(cfg=cfg, what="audio/test.tar.gz")) as resp:
        with tarfile.open(fileobj=resp, mode="r|gz") as tar:
            for member in tar:
                name = Path(member.name).name
                if not member.isfile() or name not in refs:
                    continue
                audio = read_audio(tar.extractfile(member).read())
                seconds = len(audio) / RATE
                if len(entries) < short and seconds <= 14.0:
                    path = out / f"{lang}_fleurs_{len(entries) + 1}.wav"
                    sf.write(path, audio, RATE, subtype="PCM_16")
                    entries.append({"id": path.stem, "lang": lang, "path": path.name, "seconds": round(seconds, 3),
                                    "reference": refs[name], "source": f"FLEURS {cfg} test {name}"})
                elif long_len < long_seconds:
                    long_parts += [audio, np.zeros(RATE // 2, np.float32)]
                    long_refs.append(refs[name])
                    long_len += seconds + 0.5
                if len(entries) >= short and long_len >= long_seconds:
                    break
    if long_parts:
        audio = np.concatenate(long_parts[:-1])
        path = out / f"{lang}_fleurs_long.wav"
        sf.write(path, audio, RATE, subtype="PCM_16")
        entries.append({"id": path.stem, "lang": lang, "path": path.name, "seconds": round(len(audio) / RATE, 3),
                        "reference": " ".join(long_refs), "source": f"FLEURS {cfg} test, {len(long_refs)} utterances joined"})
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--langs", nargs="+", default=["te", "ta", "hi"])
    parser.add_argument("--short", type=int, default=3)
    parser.add_argument("--long-seconds", type=float, default=90.0)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else []
    for lang in args.langs:
        manifest = [m for m in manifest if not m["id"].startswith(f"{lang}_fleurs")]
        got = fetch(lang, args.out, args.short, args.long_seconds)
        print(f"{lang}: {len(got)} clips, {sum(g['seconds'] for g in got):.1f} s")
        manifest += got
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

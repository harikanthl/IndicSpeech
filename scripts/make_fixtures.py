#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Write the small test fixtures the Swift package's tests use (a few hundred KB, no model weights).

From a reference built by export_reference.py (and the NeMo checkpoint, for the mel of a short slice):
  assets/          preprocessor.json/.bin, plus vocab.json, language_columns.json and assets.json cut down to --langs
  mel_<lang>.wav   the first --mel-seconds of one clip per language, 16-bit 16 kHz mono
  mel_<lang>.f32   NeMo's mel of exactly that wav, [80, F] little-endian float32 (F in fixtures.json)
  ctc_<lang>.f16   the clip's full-pass raw logits restricted to the language's 257 columns, [T, 257] float16
  fixtures.json    shapes, NeMo's CTC text, our frame-exact words, and NeMo's own word offsets for each clip

Usage: python make_fixtures.py --nemo m.nemo --ref ref --clips clips --out Tests/IndicSpeechTests/Fixtures
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nemo", type=Path, required=True)
    parser.add_argument("--ref", type=Path, required=True)
    parser.add_argument("--clips", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--langs", nargs="+", default=["te", "ta", "hi"])
    parser.add_argument("--mel-seconds", type=float, default=3.0)
    args = parser.parse_args()
    from export_reference import load_model

    model = load_model(args.nemo)
    out = args.out
    (out / "assets").mkdir(parents=True, exist_ok=True)
    src = args.ref / "assets"
    for name in ("preprocessor.json", "preprocessor.bin"):
        shutil.copy(src / name, out / "assets" / name)
    info = json.loads((src / "assets.json").read_text())
    info["languages"] = args.langs
    (out / "assets" / "assets.json").write_text(json.dumps(info, indent=1))
    for name in ("vocab.json", "language_columns.json"):
        full = json.loads((src / name).read_text())
        (out / "assets" / name).write_text(json.dumps({k: full[k] for k in args.langs}, ensure_ascii=False))

    manifest = json.loads((args.ref / "manifest.json").read_text())
    columns = json.loads((src / "language_columns.json").read_text())
    fixtures = {}
    for lang in args.langs:
        clip = next(c for c in manifest if c["lang"] == lang and not c["id"].endswith("long"))
        audio, rate = sf.read(args.clips / clip["path"], dtype="int16")
        cut = audio[: int(args.mel_seconds * rate)]
        sf.write(out / f"mel_{lang}.wav", cut, rate, subtype="PCM_16")
        x = torch.from_numpy(cut.astype(np.float32) / 32768.0)[None]
        with torch.inference_mode():
            mel, mel_len = model.preprocessor(input_signal=x, length=torch.tensor([x.shape[1]]))
        mel = mel[0, :, : int(mel_len[0])].numpy().astype("<f4")
        mel.tofile(out / f"mel_{lang}.f32")
        logits = np.load(args.ref / "clips" / clip["id"] / "logits.npy")[:, columns[lang]].astype("<f2")
        logits.tofile(out / f"ctc_{lang}.f16")
        decode = json.loads((args.ref / "clips" / clip["id"] / "result.json").read_text())["decodes"][lang]
        fixtures[lang] = {"clip": clip["id"], "mel_frames": int(mel.shape[1]), "mel_samples": int(len(cut)),
                          "ctc_frames": int(logits.shape[0]), "nemo_ctc_text": decode["nemo_ctc_text"],
                          "greedy_words": decode["greedy_full_words"], "nemo_ctc_words": decode.get("nemo_ctc_words")}
    (out / "fixtures.json").write_text(json.dumps(fixtures, ensure_ascii=False, indent=1))
    total = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"fixtures: {total / 1024:.0f} KB in {out}")


if __name__ == "__main__":
    main()

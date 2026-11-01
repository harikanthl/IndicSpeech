#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Side-by-side text for judging by ear and eye: NeMo (CTC, RNNT) vs each Core ML build, with word timings.

For each clip in the reference (export_reference.py) it runs the Core ML builds on the reference's own mel
windows, decodes with the clip's language, and writes one text file with:
  - the transcript from NeMo RNNT, NeMo CTC (single pass), Torch on the app's windows, and each Core ML build
  - a word table: each Core ML build's words with start–end times next to the Torch windows' words
  - WER/CER of every line against `--truth` (a text file of what was actually said) or the manifest's reference

Usage: python side_by_side.py --ref ref --models <dir> --clips te_clip_1 te_clip_2 --out telugu.txt
       [--builds pal8 fp16] [--unit ane] [--truth te_clip_1=said.txt ...]
"""
from __future__ import annotations

import argparse
import difflib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import indic_common as ic  # noqa: E402

UNITS = {"ane": "CPU_AND_NE", "gpu": "CPU_AND_GPU", "cpu": "CPU_ONLY"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ref", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--clips", nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--builds", nargs="+", default=["pal8", "fp16"])
    parser.add_argument("--unit", default="ane", choices=UNITS)
    parser.add_argument("--truth", nargs="*", default=[], help="clip_id=path/to/what-was-said.txt")
    args = parser.parse_args()
    import coremltools as ct

    truths = {k: Path(v).read_text().strip() for k, v in (t.split("=", 1) for t in args.truth)}
    manifest = {c["id"]: c for c in json.loads((args.ref / "manifest.json").read_text())}
    columns = json.loads((args.ref / "assets" / "language_columns.json").read_text())
    vocab = json.loads((args.ref / "assets" / "vocab.json").read_text())
    models = {b: ct.models.MLModel(str(args.models / f"IndicConformerCTC-{b}.mlpackage"), compute_units=getattr(ct.ComputeUnit, UNITS[args.unit]))
              for b in args.builds}
    lines: list[str] = []
    for cid in args.clips:
        clip = manifest[cid]
        result = json.loads((args.ref / "clips" / cid / "result.json").read_text())
        lang = clip["lang"] if clip["lang"] in columns else next(iter(result["decodes"]))
        d = result["decodes"][lang]
        wmel = np.load(args.ref / "clips" / cid / "window_mel.npy")
        windows = [tuple(w) for w in result["windows"]]
        texts = {"NeMo RNNT": d["nemo_rnnt_text"].strip(), "NeMo CTC (one pass)": d["nemo_ctc_text"].strip(),
                 "Torch, app windows": d["greedy_window_text"]}
        built, lid = {}, {}
        for b, model in models.items():
            logits = [model.predict({"mel": m[None], "length": np.array([v], np.int32)})["logits"][0] for m, (_, v) in zip(wmel, windows)]
            built[b] = ic.words(ic.stitch_tokens(logits, windows, np.asarray(columns[lang]), ic.word_starts(vocab[lang])), vocab[lang])
            texts[f"Core ML {b} ({args.unit})"] = ic.text_of(built[b])
            first = logits[0][: ic.encoder_frames(windows[0][1])].astype(np.float32)
            scores = []
            for code, cols in columns.items():
                sub = first[:, cols]
                lp = sub - np.logaddexp.reduce(sub, axis=1, keepdims=True)
                emit = sub.argmax(1) != ic.BLANK
                scores.append((float(lp.max(1)[emit].mean()) if emit.any() else -1e9, code))
            lid[b] = sorted(scores, reverse=True)[:3]
        truth = truths.get(cid) or clip.get("reference", "")
        lines += [f"=== {cid} ({lang}, {clip['seconds']} s, {len(windows)} windows) ===", ""]
        if truth:
            lines += [f"{'What was said':24s} {truth}", ""]
        for name, text in texts.items():
            score = ""
            if truth:
                we, wn, ce, cn = ic.error_counts(truth, text)
                score = (f"   [WER {100 * we / max(wn, 1):.1f} %, CER {100 * ce / max(cn, 1):.1f} %, "
                         f"CER without vowel signs/nukta {ic.skeleton_cer(truth, text):.1f} %]")
            lines += [f"{name:24s} {text}{score}"]
        lines += [""] + [f"Language ID, Core ML {b} (first 15.36 s, all 22 languages): " +
                         ", ".join(f"{code} {score:.3f}" for score, code in top) for b, top in lid.items()]
        lines += ["", "Word timings (s): Torch windows | " + " | ".join(f"Core ML {b}" for b in built)]
        ref_words = d["greedy_window_words"]
        for b, ws in built.items():
            sm = difflib.SequenceMatcher(a=[w["word"] for w in ref_words], b=[w["word"] for w in ws], autojunk=False)
            lines.append(f"  [{b}]")
            for tag, i1, i2, j1, j2 in sm.get_opcodes():
                if tag == "equal":
                    for r, h in zip(ref_words[i1:i2], ws[j1:j2]):
                        lines.append(f"    {r['word']:<20s} {r['start']:6.2f}–{r['end']:6.2f}  |  {h['start']:6.2f}–{h['end']:6.2f}")
                else:
                    left = ", ".join(f"{w['word']} {w['start']:.2f}–{w['end']:.2f}" for w in ref_words[i1:i2]) or "—"
                    right = ", ".join(f"{w['word']} {w['start']:.2f}–{w['end']:.2f}" for w in ws[j1:j2]) or "—"
                    lines.append(f"  ≠ {left}  |  {right}")
        lines.append("")
    args.out.write_text("\n".join(lines))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

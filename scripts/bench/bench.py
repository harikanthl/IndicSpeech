#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Parity, accuracy, language-ID and speed bench for the IndicConformer Core ML builds. macOS only.

Feeds the reference's own mel windows (export_reference.py) to each Core ML build on each compute unit, then
reports, per language and build:
  - WER/CER of Core ML vs NeMo (the same windows run in Torch, and NeMo's single full-context pass) and vs the
    datasets' ground truth (NeMo's CTC and RNNT shown alongside)
  - logits parity (cosine, max abs) and label agreement against the Torch windows
  - word timing error against the Torch windows' words (same frame rule), and NeMo's own CTC offsets vs ours
  - language ID (the first window scored under every language's mask)
  - speed: real-time factor per compute unit, compile and load times, peak memory of a process that only loads
    and runs the model

Usage: python bench.py --ref ref --models <dir with *.mlpackage> [--builds pal8 fp16] [--units ane gpu cpu]
       [--json results.json]
"""
from __future__ import annotations

import argparse
import difflib
import json
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import indic_common as ic  # noqa: E402

UNITS = {"ane": "CPU_AND_NE", "gpu": "CPU_AND_GPU", "cpu": "CPU_ONLY", "all": "ALL"}


# --- worker: one build on one compute unit, in its own process --------------------------------------------------

def worker(model_path: Path, unit: str, ref: Path, out: Path) -> None:
    import coremltools as ct

    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    t0 = time.perf_counter()
    compiled = ct.models.utils.compile_model(str(model_path))  # a fresh .mlmodelc in a temp folder
    compile_s = time.perf_counter() - t0
    cu = getattr(ct.ComputeUnit, UNITS[unit])
    t0 = time.perf_counter()
    model = ct.models.CompiledMLModel(str(compiled), compute_units=cu)
    first_load_s = time.perf_counter() - t0
    del model
    t0 = time.perf_counter()
    model = ct.models.CompiledMLModel(str(compiled), compute_units=cu)
    second_load_s = time.perf_counter() - t0
    manifest = json.loads((ref / "manifest.json").read_text())
    warm = ic.window_input(np.zeros((80, 10), np.float32), 0, 10)
    model.predict({"mel": warm, "length": np.array([10], np.int32)})
    clips, total_audio, total_compute, window_times = {}, 0.0, 0.0, []
    for clip in manifest:
        wmel = np.load(ref / "clips" / clip["id"] / "window_mel.npy")
        windows = [tuple(w) for w in json.loads((ref / "clips" / clip["id"] / "result.json").read_text())["windows"]]
        logits = []
        t0 = time.perf_counter()
        for (s, v), m in zip(windows, wmel):
            w0 = time.perf_counter()
            logits.append(model.predict({"mel": m[None], "length": np.array([v], np.int32)})["logits"][0])
            window_times.append(time.perf_counter() - w0)
        total_compute += time.perf_counter() - t0
        total_audio += clip["seconds"]
        clips[clip["id"]] = np.stack(logits).astype(np.float16)
    np.savez(out / "logits.npz", **clips)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # bytes on macOS
    (out / "speed.json").write_text(json.dumps({
        "compile_s": compile_s, "first_load_s": first_load_s, "second_load_s": second_load_s,
        "audio_s": total_audio, "compute_s": total_compute, "rtfx": total_audio / total_compute,
        "window_ms_median": 1000 * float(np.median(window_times)), "windows": len(window_times),
        "peak_rss_mb": rss / 2**20, "rss_before_load_mb": rss0 / 2**20}))


# --- analysis ------------------------------------------------------------------------------------------------------

def match_times(ref_words: list[dict], hyp_words: list[dict]) -> list[tuple[float, float]]:
    """(start diff, end diff) in seconds for words matched by text in order."""
    sm = difflib.SequenceMatcher(a=[w["word"] for w in ref_words], b=[w["word"] for w in hyp_words], autojunk=False)
    out = []
    for a, b, n in sm.get_matching_blocks():
        for k in range(n):
            r, h = ref_words[a + k], hyp_words[b + k]
            out.append((h["start"] - r["start"], h["end"] - r["end"]))
    return out


def lid(logits_window: np.ndarray, columns: dict, frames: int) -> list[tuple[str, float]]:
    scores = []
    for lang, cols in columns.items():
        sub = logits_window[:frames][:, cols].astype(np.float32)
        lp = sub - np.logaddexp.reduce(sub, axis=1, keepdims=True)
        lab = sub.argmax(1)
        emit = lab != ic.BLANK
        scores.append((lang, float(lp.max(1)[emit].mean()) if emit.any() else -1e9))
    return sorted(scores, key=lambda x: -x[1])


def rates(pairs: list[tuple[str, str]]) -> tuple[float, float]:
    we = wn = ce = cn = 0
    for ref, hyp in pairs:
        a, b, c, d = ic.error_counts(ref, hyp)
        we, wn, ce, cn = we + a, wn + b, ce + c, cn + d
    return 100 * we / max(wn, 1), 100 * ce / max(cn, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ref", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--builds", nargs="+", default=["pal8", "fp16"])
    parser.add_argument("--units", nargs="+", default=["ane", "gpu", "cpu"])
    parser.add_argument("--work", type=Path, default=Path("bench-work"))
    parser.add_argument("--json", type=Path)
    parser.add_argument("--reuse", action="store_true", help="skip runs whose output already exists")
    parser.add_argument("--worker", nargs=3, metavar=("MODEL", "UNIT", "OUT"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(Path(args.worker[0]), args.worker[1], args.ref, Path(args.worker[2]))
        return

    ref = args.ref
    manifest = json.loads((ref / "manifest.json").read_text())
    columns = json.loads((ref / "assets" / "language_columns.json").read_text())
    vocab = json.loads((ref / "assets" / "vocab.json").read_text())
    results = {c["id"]: json.loads((ref / "clips" / c["id"] / "result.json").read_text()) for c in manifest}

    speed, runs = {}, {}
    for build in args.builds:
        for unit in args.units:
            out = args.work / f"{build}-{unit}"
            out.mkdir(parents=True, exist_ok=True)
            if not (args.reuse and (out / "speed.json").exists()):
                print(f"running {build} on {unit} …", flush=True)
                subprocess.run([sys.executable, __file__, "--ref", str(ref), "--models", str(args.models),
                                "--worker", str(args.models / f"IndicConformerCTC-{build}.mlpackage"), unit, str(out)], check=True)
            speed[(build, unit)] = json.loads((out / "speed.json").read_text())
            runs[(build, unit)] = dict(np.load(out / "logits.npz"))

    report: dict = {"speed": {f"{b}/{u}": v for (b, u), v in speed.items()}, "accuracy": {}, "parity": {}, "lid": {}, "timing": {}}
    langs = sorted({c["lang"] for c in manifest if c["lang"] in columns})
    print("\n## Speed (whole test set through the model, mel windows from the reference)\n")
    print("| build | unit | RTFx | median window ms | compile s | first load s | second load s | peak RSS MB |")
    print("|---|---|---|---|---|---|---|---|")
    for (b, u), s in speed.items():
        print(f"| {b} | {u} | {s['rtfx']:.0f}× | {s['window_ms_median']:.1f} | {s['compile_s']:.1f} | {s['first_load_s']:.1f} | {s['second_load_s']:.2f} | {s['peak_rss_mb']:.0f} |")

    print("\n## Logits parity vs Torch on the same windows\n")
    print("| build | unit | cosine (min over windows) | max abs | frame label agreement (clip language) |")
    print("|---|---|---|---|---|")
    for (b, u), logits in runs.items():
        cos_min, max_abs, agree, total = 1.0, 0.0, 0, 0
        for c in manifest:
            torch_w = np.load(ref / "clips" / c["id"] / "window_logits.npy").astype(np.float32)
            ml = logits[c["id"]].astype(np.float32)
            for (s, v), tw, mw in zip(results[c["id"]]["windows"], torch_w, ml):
                n = ic.encoder_frames(v)
                x, y = tw[:n].ravel(), mw[:n].ravel()
                cos_min = min(cos_min, float(x @ y / np.linalg.norm(x) / np.linalg.norm(y)))
                max_abs = max(max_abs, float(np.abs(x - y).max()))
                if c["lang"] in columns:
                    cols = columns[c["lang"]]
                    agree += int((tw[:n][:, cols].argmax(1) == mw[:n][:, cols].argmax(1)).sum())
                    total += n
        report["parity"][f"{b}/{u}"] = {"cosine_min": cos_min, "max_abs": max_abs, "label_agreement": agree / max(total, 1)}
        print(f"| {b} | {u} | {cos_min:.5f} | {max_abs:.3f} | {100 * agree / max(total, 1):.2f} % |")

    print("\n## Accuracy per language (WER / CER %, FLEURS test clips)\n")
    print("| language | clips | audio s | NeMo RNNT vs truth | NeMo CTC vs truth | Torch windows vs truth | " +
          " | ".join(f"{b} vs truth | {b} vs Torch windows | {b} vs NeMo CTC" for b in args.builds) + " |")
    print("|---|---|---|---|---|---|" + "---|---|---|" * len(args.builds))
    unit0 = args.units[0]
    for lang in langs:
        cs = [c for c in manifest if c["lang"] == lang]
        truth = [c["reference"] for c in cs]
        d = [results[c["id"]]["decodes"][lang] for c in cs]
        row = {"rnnt_truth": rates(list(zip(truth, [x["nemo_rnnt_text"] for x in d]))),
               "ctc_truth": rates(list(zip(truth, [x["nemo_ctc_text"] for x in d]))),
               "torchwin_truth": rates(list(zip(truth, [x["greedy_window_text"] for x in d])))}
        cells = []
        for b in args.builds:
            hyps = []
            for c in cs:
                ws = ic.words(ic.stitch_tokens(list(runs[(b, unit0)][c["id"]].astype(np.float32)), results[c["id"]]["windows"],
                                               np.asarray(columns[lang]), ic.word_starts(vocab[lang])), vocab[lang])
                hyps.append(ic.text_of(ws))
                diffs = match_times(results[c["id"]]["decodes"][lang]["greedy_window_words"], ws)
                report["timing"].setdefault(f"{b}/{lang}", []).extend(diffs)
            row[f"{b}_truth"] = rates(list(zip(truth, hyps)))
            row[f"{b}_torchwin"] = rates(list(zip([x["greedy_window_text"] for x in d], hyps)))
            row[f"{b}_nemoctc"] = rates(list(zip([x["nemo_ctc_text"] for x in d], hyps)))
            cells.append(" | ".join(f"{row[k][0]:.1f} / {row[k][1]:.1f}" for k in (f"{b}_truth", f"{b}_torchwin", f"{b}_nemoctc")))
        report["accuracy"][lang] = row
        print(f"| {lang} | {len(cs)} | {sum(c['seconds'] for c in cs):.0f} | " +
              " | ".join(f"{row[k][0]:.1f} / {row[k][1]:.1f}" for k in ("rnnt_truth", "ctc_truth", "torchwin_truth")) +
              " | " + " | ".join(cells) + " |")

    print("\n## Word timing (seconds) — Core ML vs Torch windows, words matched by text\n")
    print("| build/lang | words | start mean abs | start max abs | end mean abs | end max abs |")
    print("|---|---|---|---|---|---|")
    for key, diffs in report["timing"].items():
        a = np.abs(np.asarray(diffs)) if diffs else np.zeros((1, 2))
        print(f"| {key} | {len(diffs)} | {a[:, 0].mean():.3f} | {a[:, 0].max():.2f} | {a[:, 1].mean():.3f} | {a[:, 1].max():.2f} |")
        report["timing"][key] = {"words": len(diffs), "start_mean_abs": float(a[:, 0].mean()), "start_max_abs": float(a[:, 0].max()),
                                 "end_mean_abs": float(a[:, 1].mean()), "end_max_abs": float(a[:, 1].max())}
    print("\n## NeMo 1.23's own CTC word offsets vs our frame rule (same Torch logits)\n")
    print("| language | words | start mean abs s | end mean abs s | NeMo offsets error |")
    print("|---|---|---|---|---|")
    for lang in langs:
        diffs, errors = [], set()
        for c in [c for c in manifest if c["lang"] == lang]:
            dd = results[c["id"]]["decodes"][lang]
            if "nemo_ctc_words" not in dd:
                errors.add(dd.get("nemo_ctc_words_error", "missing")[:60])
                continue
            nemo = [{"word": w["word"], "start": w["start_offset"] * ic.FRAME_SECONDS, "end": w["end_offset"] * ic.FRAME_SECONDS} for w in dd["nemo_ctc_words"]]
            diffs += match_times(nemo, dd["greedy_full_words"])
        a = np.abs(np.asarray(diffs)) if diffs else np.full((1, 2), np.nan)
        print(f"| {lang} | {len(diffs)} | {a[:, 0].mean():.3f} | {a[:, 1].mean():.3f} | {'; '.join(errors) or '—'} |")

    print("\n## Language ID (first window, all languages' masks)\n")
    for b in args.builds:
        correct = total = 0
        lines = []
        for c in manifest:
            r = results[c["id"]]
            v = r["windows"][0][1]
            ranked = lid(runs[(b, unit0)][c["id"]][0], columns, ic.encoder_frames(v))
            ok = ranked[0][0] == c["lang"]
            if c["lang"] in columns:
                correct += ok
                total += 1
            lines.append(f"  {c['id']}: " + ", ".join(f"{l} {s:.3f}" for l, s in ranked[:3]) + ("" if c["lang"] not in columns else (" ✓" if ok else " ✗")))
            report["lid"].setdefault(b, {})[c["id"]] = ranked[:5]
        print(f"{b}: {correct}/{total} correct on te/ta/hi")
        print("\n".join(lines))

    if args.json:
        args.json.write_text(json.dumps(report, indent=1, ensure_ascii=False, default=float))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build the NeMo reference for the IndicConformer 600M multilingual CTC pipeline, and dump the runtime assets.

Runs on CPU with AI4Bharat's NeMo fork (nemo-v2). For the checkpoint it writes, under <out>/assets/:
  preprocessor.json + preprocessor.bin   mel constants and the checkpoint's own Hann window and Slaney filterbank
  vocab.json                             {lang: [256 SentencePiece pieces]} in token-id order
  language_columns.json                  {lang: [257 column indices into the 5,633 logits]}; the last is the blank
  model_config.yaml, assets.json         the checkpoint's config and the shapes the runtime relies on

and for every clip in <clips>/manifest.json, under <out>/clips/<id>/:
  mel.npy           [80, F] float32 features from the fork's preprocessor (eval mode, no dither)
  logits.npy        [T, 5633] float16 raw CTC logits (decoder_layers only, before the mask and log_softmax), one
                    pass over the whole clip (full context)
  window_mel.npy    [W, 80, 1536] float32 each window's own mel (its audio featurised alone, zero-padded)
  window_logits.npy [W, 192, 5633] float16 raw logits per 15.36 s window, from the same windows the app uses
                    (indic_common.plan_windows), for Core ML parity on identical inputs
  result.json       NeMo's own CTC and RNNT text (model.transcribe with language_id), NeMo's CTC word offsets,
                    and our frame-exact greedy words decoded from logits.npy and from window_logits.npy

Usage: python export_reference.py --nemo indicconformer_stt_multi_hybrid_rnnt_600m.nemo --clips clips --out ref
"""
from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import indic_common as ic  # noqa: E402


def load_model(nemo: Path):
    from nemo.collections.asr.models import EncDecHybridRNNTCTCBPEModel

    model = EncDecHybridRNNTCTCBPEModel.restore_from(str(nemo), map_location="cpu")
    model.eval()
    model.preprocessor.featurizer.dither = 0.0
    return model


def dump_assets(model, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    fz = model.preprocessor.featurizer
    window = fz.window.detach().float().numpy()
    fb = fz.fb.detach().float().numpy()[0]  # [n_mels, bins]
    header = {"sample_rate": int(model.cfg.preprocessor.sample_rate), "n_fft": int(fz.n_fft),
              "hop_length": int(fz.hop_length), "win_length": int(fz.win_length), "n_mels": int(fb.shape[0]),
              "preemphasis": float(fz.preemph), "log_zero_guard": float(fz.log_zero_guard_value_fn(torch.zeros(1))),
              "normalize_guard": 1e-5, "normalize": str(fz.normalize), "mag_power": float(fz.mag_power)}
    (out / "preprocessor.json").write_text(json.dumps(header, indent=1))
    np.concatenate([window, fb.reshape(-1)]).astype("<f4").tofile(out / "preprocessor.bin")

    masks = model.ctc_decoder.language_masks
    langs = list(masks.keys())
    columns = {lang: np.flatnonzero(np.asarray(masks[lang], dtype=bool)).tolist() for lang in langs}
    vocab = {}
    for lang in langs:
        sp = model.tokenizer.tokenizers_dict[lang]
        pieces = sp.tokenizer if hasattr(sp, "tokenizer") else sp
        vocab[lang] = [pieces.id_to_piece(i) for i in range(pieces.get_piece_size())]
    (out / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False))
    (out / "language_columns.json").write_text(json.dumps(columns))
    from omegaconf import OmegaConf
    (out / "model_config.yaml").write_text(OmegaConf.to_yaml(model.cfg))
    info = {"languages": langs, "num_classes": int(model.ctc_decoder._num_classes),
            "columns_per_language": len(columns[langs[0]]), "blank_column": ic.BLANK,
            "vocab_sizes": {k: len(v) for k, v in vocab.items()},
            "contiguous_blocks": all(c[:-1] == list(range(c[0], c[0] + len(c) - 1)) for c in columns.values()),
            "blank_shared": len({c[-1] for c in columns.values()}) == 1,
            "window_mel_frames": ic.WINDOW_MEL_FRAMES, "window_encoder_frames": ic.encoder_frames(ic.WINDOW_MEL_FRAMES),
            "frame_seconds": ic.FRAME_SECONDS, "preprocessor": header}
    (out / "assets.json").write_text(json.dumps(info, indent=1))
    return {"langs": langs, "columns": columns, "vocab": vocab}


@torch.inference_mode()
def raw_logits(model, mel: torch.Tensor, length: torch.Tensor) -> np.ndarray:
    enc, _ = model.encoder(audio_signal=mel, length=length)
    return model.ctc_decoder.decoder_layers(enc).transpose(1, 2).float().numpy()


def nemo_texts(model, wav: Path, lang: str) -> dict:
    out = {}
    for decoder in ("ctc", "rnnt"):
        model.cur_decoder = decoder
        t0 = time.time()
        res = model.transcribe([str(wav)], batch_size=1, language_id=lang)
        res = res[0] if isinstance(res, tuple) else res
        out[f"nemo_{decoder}_text"] = res[0] if isinstance(res[0], str) else res[0].text
        out[f"nemo_{decoder}_seconds"] = round(time.time() - t0, 2)
    # NeMo's own CTC word offsets (frames), through its decoding strategy with timestamps on.
    try:
        cfg = copy.deepcopy(model.cfg.aux_ctc.decoding)
        cfg.compute_timestamps = True
        cfg.ctc_timestamp_type = "all"
        model.change_decoding_strategy(cfg, decoder_type="ctc", lang_id=lang)
        model.cur_decoder = "ctc"
        hyps = model.transcribe([str(wav)], batch_size=1, language_id=lang, return_hypotheses=True)
        hyps = hyps[0] if isinstance(hyps, tuple) else hyps
        out["nemo_ctc_words"] = [{"word": w["word"], "start_offset": int(w["start_offset"]), "end_offset": int(w["end_offset"])}
                                 for w in hyps[0].timestep["word"]]
        out["nemo_ctc_timestamp_text"] = hyps[0].text
    except Exception as error:  # recorded, not fatal: the frame-exact words below are the timing reference
        out["nemo_ctc_words_error"] = f"{type(error).__name__}: {error}"
    finally:
        cfg = copy.deepcopy(model.cfg.aux_ctc.decoding)
        cfg.compute_timestamps = False
        model.change_decoding_strategy(cfg, decoder_type="ctc", lang_id=lang)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nemo", type=Path, required=True)
    parser.add_argument("--clips", type=Path, required=True, help="folder with manifest.json and the wavs")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--only", nargs="*", help="clip ids to (re)do")
    parser.add_argument("--extra-langs", nargs="*", default=["hi", "te"],
                        help="also decode clips whose language is not in the model (e.g. English) with these masks")
    args = parser.parse_args()
    torch.set_num_threads(max(1, torch.get_num_threads()))

    model = load_model(args.nemo)
    assets = dump_assets(model, args.out / "assets")
    print("languages:", " ".join(assets["langs"]), flush=True)
    manifest = json.loads((args.clips / "manifest.json").read_text())
    shutil.copy(args.clips / "manifest.json", args.out / "manifest.json")

    for clip in manifest:
        if args.only and clip["id"] not in args.only:
            continue
        t0 = time.time()
        # model.transcribe() ends by unfreeze()-ing the encoder and decoders, which puts them back in training
        # mode (dropout on); every raw pass must run in eval mode.
        model.eval()
        wav = args.clips / clip["path"]
        audio, rate = sf.read(wav, dtype="float32")
        assert rate == 16000 and audio.ndim == 1, f"{wav}: need 16 kHz mono"
        cdir = args.out / "clips" / clip["id"]
        cdir.mkdir(parents=True, exist_ok=True)
        with torch.inference_mode():
            mel, mel_len = model.preprocessor(input_signal=torch.from_numpy(audio)[None], length=torch.tensor([len(audio)]))
        mel_np = mel[0, :, : int(mel_len[0])].numpy()
        np.save(cdir / "mel.npy", mel_np)
        full = raw_logits(model, mel, mel_len)[0]
        np.save(cdir / "logits.npy", full.astype(np.float16))
        windows = ic.plan_windows(mel_np.shape[1])
        wmels = []
        for s, v in windows:
            a, b = ic.window_samples(s, v)
            with torch.inference_mode():
                wm, wlen = model.preprocessor(input_signal=torch.from_numpy(audio[a:b])[None], length=torch.tensor([b - a]))
            assert int(wlen[0]) == v
            wmels.append(ic.window_input(wm[0, :, :v].numpy())[0])
        np.save(cdir / "window_mel.npy", np.stack(wmels))
        wl = [raw_logits(model, torch.from_numpy(m[None]), torch.tensor([v]))[0] for m, (_, v) in zip(wmels, windows)]
        np.save(cdir / "window_logits.npy", np.stack(wl).astype(np.float16))

        langs = [clip["lang"]] if clip["lang"] in assets["langs"] else args.extra_langs
        result = {"id": clip["id"], "lang": clip["lang"], "seconds": clip["seconds"], "mel_frames": int(mel_np.shape[1]),
                  "encoder_frames": int(full.shape[0]), "windows": windows, "decodes": {}}
        for lang in langs:
            cols = np.asarray(assets["columns"][lang])
            vocab = assets["vocab"][lang]
            labels, _ = ic.stitch_labels([full], [(0, mel_np.shape[1])], cols)
            ours_full = ic.words(ic.collapse(labels), vocab)
            ours_win = ic.words(ic.stitch_tokens(wl, windows, cols, ic.word_starts(vocab)), vocab)
            entry = {"greedy_full_words": ours_full, "greedy_full_text": ic.text_of(ours_full),
                     "greedy_window_words": ours_win, "greedy_window_text": ic.text_of(ours_win)}
            entry["nemo_rule_words"] = ic.nemo_rule_words(labels, vocab)
            entry.update(nemo_texts(model, wav, lang))
            model.eval()
            result["decodes"][lang] = entry
            print(f"  [{lang}] nemo ctc: {entry['nemo_ctc_text'][:90]}", flush=True)
            print(f"  [{lang}] ours    : {entry['greedy_full_text'][:90]}", flush=True)
        (cdir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=1))
        print(f"{clip['id']}: {clip['seconds']} s, {len(windows)} windows, {time.time() - t0:.1f} s", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# The trace-and-convert recipe follows FluidInference möbius's export-ctc-ja.py and quantize_coreml.py
# (https://github.com/FluidInference/mobius, Apache License 2.0): fixed shapes, torch.jit.trace, mlprogram,
# CPU_ONLY at export with compute units chosen at load, raw logits out, weights compressed after conversion.
"""Convert IndicConformer 600M multilingual (encoder + shared CTC head) to Core ML.

Writes to <out>/:
  IndicConformerCTC-fp16.mlpackage   fp16 weights and compute (~1.2 GB)
  IndicConformerCTC-pal8.mlpackage   the same, weights 8-bit k-means palettized (~0.6 GB)
  conversion.json                    versions, shapes, trace check, timings

Inputs:  mel [1, 80, 1536] float32 (normalised log-mel, see IndicMelFrontend), length [1] int32 (valid frames).
Output:  logits [1, 192, 5633] float32, raw (no mask, no log_softmax).

Runs on Linux or macOS on CPU. Usage:
  python convert_coreml.py --nemo indicconformer_stt_multi_hybrid_rnnt_600m.nemo --out build [--skip-fp16-save]
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import coremltools as ct
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import indic_common as ic  # noqa: E402
from coreml_components import EncoderCTCWrapper, rescale_residual_stream  # noqa: E402

AUTHOR = "Converted from AI4Bharat IndicConformer 600M multilingual (MIT) with scripts/indic"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nemo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--nbits", type=int, default=8)
    parser.add_argument("--kmeans-workers", type=int, default=4)
    parser.add_argument("--mel-frames", type=int, default=ic.WINDOW_MEL_FRAMES)
    parser.add_argument("--fp32-ops", default="",
                        help="comma-separated MIL op types kept in fp32 (e.g. softmax,layer_norm); off-Neural-Engine ops")
    parser.add_argument("--fp16-only", action="store_true", help="save the fp16 build only, skip palettization")
    parser.add_argument("--name", default="IndicConformerCTC", help="output package name prefix")
    parser.add_argument("--stream-scale", type=float, default=1.0,
                        help="residual-stream rescale (exact up to the norms' epsilon); 1/16 was tried for the Neural "
                             "Engine's fp16 drift and made no difference (2026-11-01), so it is off")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    log: dict = {"python": platform.python_version(), "torch": torch.__version__, "coremltools": ct.__version__,
                 "platform": platform.platform(), "mel_frames": args.mel_frames}

    from export_reference import load_model
    t0 = time.time()
    model = load_model(args.nemo)
    wrapper = EncoderCTCWrapper(model.encoder, model.ctc_decoder).eval()
    log["load_seconds"] = round(time.time() - t0, 1)

    torch.manual_seed(0)
    mel = torch.randn(1, 80, args.mel_frames)
    full = torch.tensor([args.mel_frames], dtype=torch.int32)
    part = torch.tensor([args.mel_frames * 2 // 3], dtype=torch.int32)
    if args.stream_scale != 1:
        with torch.inference_mode():
            before = wrapper(mel, full)
        log["stream_scale"] = args.stream_scale
        log["stream_scale_tensors"] = rescale_residual_stream(model.encoder, args.stream_scale)
        with torch.inference_mode():
            log["stream_scale_max_abs_change"] = float((wrapper(mel, full) - before).abs().max())
        print(f"residual stream × {args.stream_scale}: logits change {log['stream_scale_max_abs_change']:.2e}", flush=True)
    t0 = time.time()
    with torch.inference_mode():
        eager_full, eager_part = wrapper(mel, full), wrapper(mel, part)
    traced = torch.jit.trace(wrapper, (mel, full), strict=False, check_trace=False)
    with torch.inference_mode():
        traced_full, traced_part = traced(mel, full), traced(mel, part)
    log["trace_seconds"] = round(time.time() - t0, 1)
    log["output_shape"] = list(eager_full.shape)
    # The length must stay a live input in the trace (it masks attention and convolutions for padded windows).
    log["trace_max_abs_diff_full"] = float((eager_full - traced_full).abs().max())
    log["trace_max_abs_diff_partial_length"] = float((eager_part - traced_part).abs().max())
    log["partial_length_changes_output"] = float((eager_full - eager_part).abs().max())
    print(json.dumps(log, indent=1), flush=True)
    assert log["trace_max_abs_diff_partial_length"] < 1e-3, "length is not traced as an input"

    t0 = time.time()
    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[ct.TensorType(name="mel", shape=(1, 80, args.mel_frames), dtype=np.float32),
                ct.TensorType(name="length", shape=(1,), dtype=np.int32)],
        outputs=[ct.TensorType(name="logits", dtype=np.float32)],
        compute_units=ct.ComputeUnit.CPU_ONLY,
        compute_precision=(ct.transform.FP16ComputePrecision(op_selector=lambda op, keep=set(filter(None, args.fp32_ops.split(","))): op.op_type not in keep)
                           if args.fp32_ops else ct.precision.FLOAT16),
        minimum_deployment_target=ct.target.iOS18,
    )
    log["convert_seconds"] = round(time.time() - t0, 1)
    meta = {"frame_seconds": str(ic.FRAME_SECONDS), "window_mel_frames": str(args.mel_frames),
            "encoder_frames": str(ic.encoder_frames(args.mel_frames)), "num_classes": "5633", "blank_in_slice": "256",
            "source": "indicconformer_stt_multi_hybrid_rnnt_600m.nemo (AI4Bharat, MIT)"}
    mlmodel.author = AUTHOR
    mlmodel.license = "MIT (model weights, AI4Bharat); conversion scripts Apache-2.0"
    mlmodel.short_description = "IndicConformer 600M multilingual encoder + shared CTC head, raw logits"
    mlmodel.user_defined_metadata.update(meta)
    log["fp32_ops"] = args.fp32_ops
    path = args.out / f"{args.name}-fp16.mlpackage"
    mlmodel.save(str(path))
    print(f"saved {path} in {log['convert_seconds']} s", flush=True)

    (args.out / "conversion.json").write_text(json.dumps(log, indent=1))
    # Palettize in a fresh process: forking k-means workers from this one (torch's thread pools are up) deadlocks.
    import subprocess
    if args.fp16_only:
        return
    subprocess.run([sys.executable, str(Path(__file__).resolve().parent / "palettize.py"), str(path),
                    "--nbits", str(args.nbits), "--workers", str(args.kmeans_workers)], check=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
#
# Compression step of the möbius recipe (quantize_coreml.py, https://github.com/FluidInference/mobius, Apache-2.0):
# k-means palettization of an converted mlprogram, weights only.
"""Palettize a Core ML mlprogram's weights (k-means, per tensor) and save it next to the source.

Runs in its own process without torch: coremltools forks k-means workers, and forking after torch has started
its thread pools deadlocks them (seen on Linux: every worker at 0 % CPU).

Usage: python palettize.py IndicConformerCTC-fp16.mlpackage --nbits 8 --workers 4
       → IndicConformerCTC-pal8.mlpackage
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import coremltools as ct
from coremltools.optimize.coreml import OpPalettizerConfig, OptimizationConfig, palettize_weights


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path)
    parser.add_argument("--nbits", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    t0 = time.time()
    model = ct.models.MLModel(str(args.source), skip_model_load=True)
    config = OptimizationConfig(global_config=OpPalettizerConfig(
        mode="kmeans", nbits=args.nbits, granularity="per_tensor", weight_threshold=2048, num_kmeans_workers=args.workers))
    pal = palettize_weights(model, config)
    pal.short_description = model.short_description + f", {args.nbits}-bit palettized"
    out = args.source.with_name(args.source.name.replace("-fp16", f"-pal{args.nbits}"))
    pal.save(str(out))
    seconds = round(time.time() - t0, 1)
    print(f"saved {out} in {seconds} s", flush=True)
    log = args.source.parent / "conversion.json"
    info = json.loads(log.read_text()) if log.exists() else {}
    info[f"palettize_{args.nbits}bit_seconds"] = seconds
    log.write_text(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()

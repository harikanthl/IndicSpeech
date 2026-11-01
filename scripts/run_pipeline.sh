#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# The whole conversion-machine run: reference and conversion in parallel (each loads its own copy of the model;
# ~19 GB RAM is enough). Paths are arguments, nothing is hard-coded.
# Usage: run_pipeline.sh <model.nemo> <clips dir> <out dir>
set -euo pipefail
NEMO=$1; CLIPS=$2; OUT=$3
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$OUT/logs"
tar -tf "$NEMO" > "$OUT/logs/nemo-contents.txt"   # a truncated download fails here
python "$HERE/export_reference.py" --nemo "$NEMO" --clips "$CLIPS" --out "$OUT/ref" > "$OUT/logs/reference.log" 2>&1 &
REF=$!
python "$HERE/convert_coreml.py" --nemo "$NEMO" --out "$OUT/coreml" > "$OUT/logs/convert.log" 2>&1 &
CONV=$!
wait $REF && echo "REFERENCE_OK" || echo "REFERENCE_FAILED"
wait $CONV && echo "CONVERT_OK" || echo "CONVERT_FAILED"

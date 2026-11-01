# IndicConformer → Core ML

Converts AI4Bharat's **IndicConformer 600M multilingual** (hybrid CTC/RNNT FastConformer, 22 Indian languages,
MIT) to Core ML with its shared CTC head, builds a NeMo reference to check it against, and benchmarks the result on
a Mac. The Swift runtime that uses the output is the `IndicSpeech` package at the root of this repo. Every path here is an
argument.

What comes out:

| File | What |
|---|---|
| `IndicConformerCTC-pal8.mlpackage` | encoder + shared CTC head, weights 8-bit k-means palettized (~0.6 GB). The default build |
| `IndicConformerCTC-fp16.mlpackage` | the same in fp16 (~1.2 GB), for devices that reject palettized weights |
| `preprocessor.json`, `preprocessor.bin` | mel constants, the checkpoint's own Hann window and Slaney filterbank (float32) |
| `vocab.json` | `{language: [256 SentencePiece pieces]}` |
| `language_columns.json` | `{language: [257 columns of the 5,633 logits]}`, the last being the blank |
| `assets.json` | languages in order, head width, shapes |

Model I/O: `mel [1, 80, 1536]` float32 (normalised log-mel, 15.36 s) and `length [1]` int32 (valid frames) in,
`logits [1, 192, 5633]` float32 out: raw, one row per 80 ms encoder frame, no language mask and no log-softmax (the
host takes the language's 257 columns and an argmax). Shorter audio is zero-padded with its true length; longer
audio is cut into overlapping windows (`indic_common.plan_windows`).

## Files

| Script | Runs on | Does |
|---|---|---|
| `fetch_clips.py` | anywhere | streams a few FLEURS test utterances per language (no full download) plus a ~90 s joined clip |
| `export_reference.py` | conversion machine | NeMo reference: mel, raw logits (whole clip and per window), NeMo CTC + RNNT text, NeMo's CTC word offsets, frame-exact greedy words; dumps the runtime assets |
| `convert_coreml.py` | conversion machine | trace + `ct.convert` (fp16), then runs `palettize.py` |
| `palettize.py` | conversion machine | 8-bit k-means palettization of the fp16 package, in a torch-free process |
| `run_pipeline.sh` | conversion machine | the two above in parallel |
| `make_fixtures.py` | conversion machine | the Swift package's small test fixtures (no weights) |
| `bench/bench.py` | Mac | parity, WER/CER, timing error, language ID, speed and memory per build and compute unit |
| `indic_common.py` | both | windows, stitching, greedy CTC, words, WER/CER (numpy only); the Swift package mirrors it |
| `coreml_components.py` | conversion machine | the traceable encoder + CTC wrapper |

## 1. Conversion machine (Linux or macOS, CPU only; ~20 GB RAM, ~10 GB disk)

Tested on Ubuntu 24.04 x86_64, 4 cores, 19 GB RAM, Python 3.10.22, uv 0.13.6. One venv holds both the NeMo fork
and coremltools: the fork (1.23.0rc0) doesn't pin torch, and torch 2.7.0 is the newest coremltools 9.0 supports.

```sh
mkdir -p work && cd work
git clone --depth 1 -b nemo-v2 https://github.com/AI4Bharat/NeMo.git nemo-ai4b   # tested at 8dce88cf8e94
uv venv -p 3.10 venv && . venv/bin/activate
uv pip install -r /path/to/scripts/indic/requirements-conversion.txt \
  --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
uv pip install --no-deps -e nemo-ai4b
python -c "import nemo.collections.asr, coremltools, torch; print(torch.__version__, coremltools.__version__)"

# The checkpoint: public link from https://github.com/AI4Bharat/IndicConformerASR (2.52 GB; the HF repo is gated
# and ships ONNX, not the .nemo)
wget -c https://objectstore.e2enetworks.net/indicconformer/models/indicconformer_stt_multi_hybrid_rnnt_600m.nemo

S=/path/to/scripts/indic
python $S/fetch_clips.py --out clips --langs te ta hi --short 3 --long-seconds 90
# optional: add your own 16 kHz mono wavs to clips/manifest.json ({id, lang, path, seconds, reference})
bash $S/run_pipeline.sh indicconformer_stt_multi_hybrid_rnnt_600m.nemo clips out
python $S/make_fixtures.py --nemo indicconformer_stt_multi_hybrid_rnnt_600m.nemo --ref out/ref --clips clips \
  --out fixtures
```

`out/coreml/` then holds the two `.mlpackage`s and `conversion.json` (versions, trace check, timings);
`out/ref/assets/` the runtime assets; `out/ref/clips/<id>/` the reference per clip.

## 2. Mac: bench

```sh
uv venv -p 3.12 venv && VIRTUAL_ENV=$PWD/venv uv pip install numpy==1.26.4 coremltools==9.0 soundfile==0.12.1
mkdir model && cp -R out/coreml/*.mlpackage out/ref/assets/* model/
venv/bin/python $S/bench/bench.py --ref out/ref --models model --builds pal8 fp16 --units ane gpu cpu --json bench.json
```

The bench prints markdown tables: speed (RTFx, compile and load times, peak RSS), logits parity against Torch on
identical windows, WER/CER per language against Torch, NeMo and the ground truth, word timing error, and language
ID. The Swift package's end-to-end test runs the same clips through the real Swift pipeline:

```sh
INDIC_SPEECH_MODEL_DIR=$PWD/model INDIC_SPEECH_REF_DIR=$PWD/out/ref INDIC_SPEECH_CLIPS_DIR=$PWD/clips \
  swift test --package-path /path/to/IndicSpeech
```

## Licences

- These scripts: Apache-2.0. `coreml_components.py` and the recipe in `convert_coreml.py` are adapted from
  FluidInference möbius (https://github.com/FluidInference/mobius, Apache-2.0); see the file headers.
- The model weights, its config and tokenizers: MIT, AI4Bharat (https://github.com/AI4Bharat/IndicConformerASR).
- AI4Bharat's NeMo fork: Apache-2.0. coremltools: BSD-3-Clause.
- FLEURS test audio and transcripts: CC-BY-4.0, Google.

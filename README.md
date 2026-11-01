<p align="center">
  <img src=".github/logo.png" alt="IndicSpeech logo: a sound wave turning into a handwritten stroke" width="320">
</p>

<h1 align="center">IndicSpeech</h1>

<p align="center"><b>On-device speech recognition for 22 Indian languages on iPhone and Mac.</b><br>
తెలుగు · हिन्दी · தமிழ் · বাংলা · ಕನ್ನಡ · ગુજરાતી · മലയാളം · मराठी · ଓଡ଼ିଆ · ਪੰਜਾਬੀ · اردو …</p>

IndicSpeech runs [AI4Bharat](https://ai4bharat.iitm.ac.in/)'s
[IndicConformer 600M](https://github.com/AI4Bharat/IndicConformerASR) on the Apple Neural Engine. It turns speech in
Telugu, Hindi, Tamil and 19 more languages into text, with a start and end time for every word, and it can tell which
language is being spoken. It's free, it works offline, and the audio never leaves the device.

```swift
import IndicSpeech

let engine = try await IndicConformerEngine(modelDirectory: modelURL)
let result = try await engine.transcribe(contentsOf: audioURL, language: .telugu)
for word in result.words {
    print(word.text, word.start, word.end)   // నమస్కారం 1.60 2.24 …
}

// Not sure which language it is?
let best = try await engine.detectLanguage(contentsOf: audioURL).first?.language   // .telugu
```

Built for [ezRecord](https://github.com/harikanthl/ezRecord), where it powers Telugu and other Indian-language
captions and editing by words. Shared so any Apple app can understand India's languages.

## Languages

Assamese, Bengali, Bodo, Dogri, Gujarati, Hindi, Kannada, Kashmiri, Konkani, Maithili, Malayalam, Manipuri, Marathi,
Nepali, Odia, Punjabi, Sanskrit, Santali, Sindhi, Tamil, Telugu, Urdu. That's every one of India's 22 scheduled
languages, from one model.

## How well it works

Measured on an Apple M4 Mac:

- **Fast.** About **400× real time** on the Neural Engine, so an hour of audio takes about 9 seconds. Peak memory is
  about 0.75 GB. The first load takes 15–20 s while the Neural Engine compiles the model; later loads take about
  0.1 s.
- **Faithful to the original.** On Google's FLEURS test clips it stays within 0.5 points of word error rate of
  AI4Bharat's own PyTorch model for Hindi, Tamil and Telugu.
- **Real speech.** On a native speaker's 40-second Telugu recording, made in an ordinary room, it got every word right
  (0.0 % character error rate). Before this, Apple's built-in recognizer turned the same recording into English-letter
  guesses.
- **Language detection.** It picked the right language on 11 of 12 test clips. The miss was a 5-second Hindi clip
  that scored Urdu by a hair.

Accuracy varies by language: it's strongest on widely spoken languages and weaker on low-resource ones such as Bodo
and Dogri. See [AI4Bharat's results](https://github.com/AI4Bharat/IndicConformerASR).

## Install

Swift Package Manager, macOS 26 / iOS 26, Swift 6. It uses system frameworks only (Accelerate, AVFoundation,
Core ML).

```swift
.package(url: "https://github.com/harikanthl/IndicSpeech.git", from: "1.0.0")
```

## The model

The model isn't in this repo. It's on Hugging Face (MIT, about 600 MB):
**[harikanth/indic-conformer-600m-multilingual-coreml](https://huggingface.co/harikanth/indic-conformer-600m-multilingual-coreml)**

```sh
hf download harikanth/indic-conformer-600m-multilingual-coreml --local-dir IndicConformer
```

Point `modelDirectory` at that folder. It holds `IndicConformerCTC-pal8.mlpackage` (and a compiled `.mlmodelc`),
plus `preprocessor.json`, `preprocessor.bin`, `vocab.json`, `language_columns.json` and `assets.json`. The repo
also has a `manifest.json` with every file's size and SHA-256, for apps that download the model themselves.

## How it works

- **One model, 22 languages.** The CTC head is shared. A language is just 257 of its 5,633 output columns, chosen
  with a mask.
- **Language detection comes almost free:** decode the first window under every language's mask and keep the most
  confident one. There's no second model.
- **Mel frontend in Swift** (vDSP), matching NeMo's preprocessor to a cosine of 0.9999999999.
- **Any length of audio.** It's cut into 15.36 s windows with overlap. Each window is featurised from its own audio,
  and neighbouring windows are joined at a word both of them emit. Cutting at a fixed frame duplicates syllables,
  because CTC places tokens a few frames later when it has less audio ahead.
- **Word timings** come from the CTC frames (80 ms each). CTC word ends land slightly early; refine them with forced
  alignment if you need tight karaoke timing.
- **No punctuation:** the model writes words only. Break captions on pauses, or add a punctuation pass afterwards.
- **`indic-transcribe`** is a command-line tool for trying it on a file.

## Converting the model yourself

`scripts/` has the whole pipeline, with pinned versions and exact commands in [scripts/README.md](scripts/README.md):

1. Fetch test clips.
2. Export AI4Bharat's NeMo reference (text, logits, word timings).
3. Convert to Core ML.
4. Palettize to 8-bit.
5. Make the test fixtures.
6. Benchmark accuracy, timing, language detection, speed and memory on a Mac.

## Tests

`swift test` runs parity tests on small fixtures:
- the mel frontend against NeMo's preprocessor;
- greedy CTC and word times against NeMo's logits;
- window planning and stitching.

With `INDIC_SPEECH_MODEL_DIR`, `INDIC_SPEECH_REF_DIR` and `INDIC_SPEECH_CLIPS_DIR` set, an end-to-end test also runs
every reference clip through the real model.

## Not done yet

- iPhone speed and memory haven't been measured. Some iPhone Neural Engines reject palettized weights, so an fp16
  build is planned for them.
- There's no punctuation. AI4Bharat's Cadence is the planned add-on.
- Decoding is CTC only. RNNT decoding is 2–4 points more accurate in AI4Bharat's tests and could come later.

Contributions are welcome: issues, accuracy reports for your language, and fixes.

## Licence and credits

- **This package and scripts:** Apache-2.0 ([LICENSE](LICENSE)).
- **The model:** [AI4Bharat IndicConformer](https://github.com/AI4Bharat/IndicConformerASR), MIT. All credit for the
  model goes to AI4Bharat at IIT Madras.
- **Adapted code** (details in [NOTICE](NOTICE)):
  - the mel frontend, from [Muesli](https://github.com/Muesli-HQ/muesli), MIT;
  - the conversion scripts, from FluidInference's [möbius](https://github.com/FluidInference/mobius), Apache-2.0.
- **Test fixtures:** short excerpts from Google's [FLEURS](https://huggingface.co/datasets/google/fleurs),
  CC BY 4.0.

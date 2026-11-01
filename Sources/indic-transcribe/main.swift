// SPDX-License-Identifier: Apache-2.0
//
// indic-transcribe: the IndicSpeech engine from the command line, for trying the model on your own recordings.
//
//   swift run -c release indic-transcribe <audio file> --model <model dir> [--language te] [--build pal8|fp16]
//                                         [--units ane|gpu|cpu] [--json]
//
// Without --language it detects the language first. Prints the text, then one word per line with start and end
// seconds and confidence, then timing; --json prints the words as JSON instead.

import CoreML
import Foundation
import IndicSpeech

func value(_ flag: String) -> String? {
    let args = CommandLine.arguments
    guard let i = args.firstIndex(of: flag), i + 1 < args.count else { return nil }
    return args[i + 1]
}

let positional = CommandLine.arguments.dropFirst().filter { !$0.hasPrefix("--") }
    .filter { arg in !["--model", "--language", "--build", "--units"].contains { value($0) == arg } }
guard let path = positional.first, let modelDir = value("--model") else {
    print("usage: indic-transcribe <audio file> --model <model dir> [--language te] [--build pal8|fp16] [--units ane|gpu|cpu] [--json]")
    exit(2)
}
let units: MLComputeUnits = switch value("--units") { case "gpu": .cpuAndGPU; case "cpu": .cpuOnly; default: .cpuAndNeuralEngine }
let clock = ContinuousClock.now
let engine = try await IndicConformerEngine(modelDirectory: URL(fileURLWithPath: modelDir),
                                            modelName: "IndicConformerCTC-\(value("--build") ?? "pal8")", computeUnits: units)
let loaded = ContinuousClock.now - clock
let audio = try AudioLoader.load(URL(fileURLWithPath: path))
var language = value("--language").map(IndicLanguage.init(rawValue:))
var detected = ""
if language == nil {
    let ranked = try await engine.detectLanguage(audio: audio)
    language = ranked.first?.language
    detected = ranked.prefix(3).map { "\($0.language) \(String(format: "%.3f", $0.score))" }.joined(separator: ", ")
}
let result = try await engine.transcribe(audio: audio, language: language!)
if CommandLine.arguments.contains("--json") {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .withoutEscapingSlashes]
    print(String(decoding: try encoder.encode(result.words), as: UTF8.self))
} else {
    if !detected.isEmpty { print("language: \(detected)") }
    print(result.text, "\n")
    for w in result.words { print(String(format: "%7.2f  %7.2f  %.2f  ", w.start, w.end, w.confidence) + w.text) }
    print(String(format: "\n%.1f s of audio, %d windows, %.3f s compute (%.0f× real time), model load %@",
                 result.audioSeconds, result.windowCount, result.computeSeconds, result.audioSeconds / result.computeSeconds,
                 "\(loaded)"))
}

// SPDX-License-Identifier: Apache-2.0
//
// End to end on the real model, when it is on this machine: every reference clip's wav goes through
// IndicConformerEngine (AudioLoader → Swift mel → Core ML → stitching → greedy CTC) and is compared with the
// Torch run of the same windows (export_reference.py's greedy_window_text), and the Swift mel of the whole clip
// with NeMo's. Prints per-clip word error, real-time factor and the language ID ranking.
//
//   INDIC_SPEECH_MODEL_DIR=<model + assets> INDIC_SPEECH_REF_DIR=<ref> INDIC_SPEECH_CLIPS_DIR=<wavs> swift test
//   (optional INDIC_SPEECH_MODEL_NAME=IndicConformerCTC-fp16, INDIC_SPEECH_UNITS=ane|gpu|cpu)
//
// Without those variables the test is skipped, so the package's tests need no model.

import CoreML
import Foundation
import Testing
@testable import IndicSpeech

private let env = ProcessInfo.processInfo.environment
private let hasModel = env["INDIC_SPEECH_MODEL_DIR"] != nil && env["INDIC_SPEECH_REF_DIR"] != nil

/// Minimal .npy reader for C-order little-endian float32/float16 arrays.
func readNpy(_ url: URL) throws -> (shape: [Int], values: [Float]) {
    let data = try Data(contentsOf: url)
    let headerLength = Int(data[8]) | Int(data[9]) << 8
    let header = String(decoding: data[10..<(10 + headerLength)], as: UTF8.self)
    let shapeText = header.split(separator: "(")[1].split(separator: ")")[0]
    let shape = shapeText.split(separator: ",").compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
    let body = data.subdata(in: (10 + headerLength)..<data.count)
    let values: [Float] = body.withUnsafeBytes { raw in
        if header.contains("<f2") {
            return (0..<(body.count / 2)).map { Float(Float16(bitPattern: raw.loadUnaligned(fromByteOffset: $0 * 2, as: UInt16.self))) }
        }
        return (0..<(body.count / 4)).map { raw.loadUnaligned(fromByteOffset: $0 * 4, as: Float.self) }
    }
    return (shape, values)
}

func wordErrors(_ reference: [String], _ hypothesis: [String]) -> Int {
    var previous = Array(0...hypothesis.count)
    for (i, r) in reference.enumerated() {
        var current = [i + 1] + Array(repeating: 0, count: hypothesis.count)
        for (j, h) in hypothesis.enumerated() {
            current[j + 1] = min(previous[j + 1] + 1, current[j] + 1, previous[j] + (r == h ? 0 : 1))
        }
        previous = current
    }
    return previous[hypothesis.count]
}

@Suite(.enabled(if: hasModel, "set INDIC_SPEECH_MODEL_DIR and INDIC_SPEECH_REF_DIR to run on the real model"))
struct EngineEndToEndTests {
    struct Clip: Decodable { var id: String; var lang: String; var path: String; var seconds: Double }
    struct Decode: Decodable { var greedy_window_text: String }
    struct Result: Decodable { var decodes: [String: Decode] }

    @Test func transcribesTheReferenceClipsLikeTorch() async throws {
        let modelDir = URL(fileURLWithPath: env["INDIC_SPEECH_MODEL_DIR"]!)
        let ref = URL(fileURLWithPath: env["INDIC_SPEECH_REF_DIR"]!)
        let clipsDir = env["INDIC_SPEECH_CLIPS_DIR"].map(URL.init(fileURLWithPath:)) ?? ref.appending(path: "wav")
        let units: MLComputeUnits = switch env["INDIC_SPEECH_UNITS"] { case "gpu": .cpuAndGPU; case "cpu": .cpuOnly; default: .cpuAndNeuralEngine }
        let name = env["INDIC_SPEECH_MODEL_NAME"] ?? IndicConformerEngine.defaultModelName
        let loadClock = ContinuousClock.now
        let engine = try await IndicConformerEngine(modelDirectory: modelDir, modelName: name, computeUnits: units)
        print("engine \(name) [\(env["INDIC_SPEECH_UNITS"] ?? "ane")]: compile \(engine.model.compileSeconds) s, load \(engine.model.loadSeconds) s, total \((ContinuousClock.now - loadClock).seconds) s")

        let clips = try JSONDecoder().decode([Clip].self, from: Data(contentsOf: ref.appending(path: "manifest.json")))
        var errors = 0, words = 0, audio = 0.0, compute = 0.0
        var worstMelCosine = 1.0
        for clip in clips {
            let wav = clipsDir.appending(path: clip.path)
            let samples = try AudioLoader.load(wav)
            // Whole-clip mel parity.
            let mel = engine.frontend.features(samples)
            let refMel = try readNpy(ref.appending(path: "clips/\(clip.id)/mel.npy"))
            #expect(refMel.shape == [80, mel.frames])
            var dot = 0.0, a2 = 0.0, b2 = 0.0
            for (x, y) in zip(mel.values, refMel.values) { dot += Double(x * y); a2 += Double(x * x); b2 += Double(y * y) }
            worstMelCosine = min(worstMelCosine, dot / (a2.squareRoot() * b2.squareRoot()))

            let result = try JSONDecoder().decode(Result.self, from: Data(contentsOf: ref.appending(path: "clips/\(clip.id)/result.json")))
            let language = engine.vocabulary.supports(IndicLanguage(clip.lang)) ? IndicLanguage(clip.lang) : .hindi
            guard let expected = result.decodes[language.rawValue]?.greedy_window_text else { continue }
            let t = try await engine.transcribe(audio: samples, language: language)
            let e = wordErrors(expected.split(separator: " ").map(String.init), t.words.map(\.text))
            let n = expected.split(separator: " ").count
            errors += e; words += n; audio += t.audioSeconds; compute += t.computeSeconds
            let guess = try await engine.detectLanguage(audio: samples, candidates: nil).prefix(3)
            print("\(clip.id) [\(language)]: \(t.windowCount) windows, \(e)/\(n) words differ from Torch, \(String(format: "%.0f", t.audioSeconds / t.computeSeconds))× real time; language ID: \(guess.map { "\($0.language) \(String(format: "%.3f", $0.score))" }.joined(separator: ", "))")
        }
        print("all clips: WER vs Torch windows \(String(format: "%.2f", 100 * Double(errors) / Double(max(words, 1)))) %, \(String(format: "%.0f", audio / compute))× real time, worst whole-clip mel cosine \(worstMelCosine)")
        #expect(worstMelCosine > 0.99999)
        #expect(Double(errors) / Double(max(words, 1)) < 0.03)
    }
}

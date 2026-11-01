// SPDX-License-Identifier: Apache-2.0
//
// The package's front door: speech in, words with times out, for any of the model's 22 Indian languages.
//
//   let engine = try await IndicConformerEngine(modelDirectory: dir)
//   let result = try await engine.transcribe(contentsOf: url, language: .telugu)
//   let guess = try await engine.detectLanguage(audio: samples)
//
// Pipeline: 16 kHz mono → WindowPlanner's 15.36 s windows → IndicMelFrontend per window (its own audio,
// normalised over itself) → IndicConformerModel (Core ML, one call per window) → per-frame argmax over the
// language's 257 columns → collapse per window → WindowStitcher (hand-over at a word both windows emit) →
// CTCGreedyDecoder.words (80 ms frame times).
//
// The model directory holds what scripts/indic produces: the .mlpackage (or compiled .mlmodelc) plus
// preprocessor.json/.bin, vocab.json, language_columns.json and assets.json.
//
// Language ID comes free from the shared head: every language's slice of the same logits is a full
// distribution, so the first window's logits are scored under each language's mask and the best wins (see
// LanguageScorer). The engine is an actor; windows run one after another (the Neural Engine runs one at a time
// anyway), and nothing touches the main thread.

import AVFoundation
import CoreML
import Foundation

public struct IndicTranscription: Sendable, Equatable {
    public var language: IndicLanguage
    public var words: [IndicWord]
    public var audioSeconds: Double
    /// Wall time for mel + model + decode, excluding audio file decoding.
    public var computeSeconds: Double
    public var windowCount: Int
    public var text: String { words.map(\.text).joined(separator: " ") }

    public init(language: IndicLanguage, words: [IndicWord], audioSeconds: Double, computeSeconds: Double, windowCount: Int) {
        self.language = language; self.words = words; self.audioSeconds = audioSeconds
        self.computeSeconds = computeSeconds; self.windowCount = windowCount
    }
}

public actor IndicConformerEngine {
    public static let sampleRate = 16_000.0
    public static let defaultModelName = "IndicConformerCTC-pal8"

    public nonisolated let vocabulary: IndicVocabulary
    public nonisolated let frontend: IndicMelFrontend
    public nonisolated let model: IndicConformerModel

    public nonisolated var languages: [IndicLanguage] { vocabulary.languages }

    /// Loads the model files from `modelDirectory`. The first load of an `.mlpackage` compiles it (seconds).
    public init(modelDirectory: URL, modelName: String = IndicConformerEngine.defaultModelName,
                computeUnits: MLComputeUnits = .cpuAndNeuralEngine, compiledDirectory: URL? = nil) async throws {
        vocabulary = try IndicVocabulary.load(from: modelDirectory)
        frontend = try IndicMelFrontend(constants: IndicPreprocessorConstants.load(from: modelDirectory))
        let compiled = modelDirectory.appending(path: modelName + ".mlmodelc")
        let package = modelDirectory.appending(path: modelName + ".mlpackage")
        let url = FileManager.default.fileExists(atPath: package.path) ? package : compiled
        model = try await IndicConformerModel(url: url, computeUnits: computeUnits, compiledDirectory: compiledDirectory)
        guard model.classCount == vocabulary.classCount, model.windowFrames == WindowPlanner.windowFrames else {
            throw IndicSpeechError.badModelFiles("model shape [\(model.windowFrames) → \(model.encoderFrames) × \(model.classCount)] doesn't match the vocabulary or the window plan")
        }
    }

    // MARK: Transcription

    public func transcribe(contentsOf url: URL, language: IndicLanguage) async throws -> IndicTranscription {
        try await transcribe(audio: AudioLoader.load(url), language: language)
    }

    /// `audio` is 16 kHz mono in −1…1.
    public func transcribe(audio: [Float], language: IndicLanguage) async throws -> IndicTranscription {
        let clock = ContinuousClock.now
        let columns = try vocabulary.columns(for: language)
        let pieces = try vocabulary.pieces(for: language)
        guard audio.count >= 2 * frontend.constants.hopLength else {
            return IndicTranscription(language: language, words: [], audioSeconds: Double(audio.count) / Self.sampleRate,
                                      computeSeconds: 0, windowCount: 0)
        }
        // Each window is featurised from its own audio and normalised over itself (as NeMo's buffered inference
        // does); one normalisation over a long clip left whole windows blank.
        let windows = WindowPlanner.plan(melFrames: frontend.frameCount(samples: audio.count))
        let wordStart = pieces.map { $0.hasPrefix("\u{2581}") } + [false]
        var perWindow: [[CTCToken]] = []
        for window in windows {
            try Task.checkCancellation()
            let mel = frontend.features(Array(audio[Self.samples(of: window, total: audio.count)]))
            let logits = try await model.logits(mel: mel, start: 0, valid: window.validFrames)
            let frames = WindowPlanner.encoderFrames(window.validFrames)
            let labels = logits.withUnsafeBufferPointer {
                CTCGreedyDecoder.labels(logits: $0, frames: 0..<frames, classCount: model.classCount, columns: columns)
            }
            perWindow.append(CTCGreedyDecoder.collapse(labels, blank: vocabulary.blank).map {
                var t = $0; t.firstFrame += window.encoderOffset; t.lastFrame += window.encoderOffset; return t
            })
        }
        let tokens = WindowStitcher.merge(perWindow, windows: windows, wordStart: wordStart)
        return IndicTranscription(language: language, words: CTCGreedyDecoder.words(tokens, pieces: pieces),
                                  audioSeconds: Double(audio.count) / Self.sampleRate,
                                  computeSeconds: (ContinuousClock.now - clock).seconds, windowCount: windows.count)
    }

    /// The samples whose centred mel frames are exactly the window's (hop 160).
    static func samples(of window: EncoderWindow, total: Int) -> Range<Int> {
        let from = window.start * 160
        return from..<min(total, (window.start + window.validFrames - 1) * 160 + 1)
    }

    // MARK: Language ID

    /// Scores every language (or `candidates`) on the first 15.36 s window of `audio`, best first.
    public func detectLanguage(audio: [Float], candidates: [IndicLanguage]? = nil) async throws -> [LanguageScore] {
        let mel = frontend.features(Array(audio.prefix((WindowPlanner.windowFrames - 1) * frontend.constants.hopLength + 1)))
        let window = WindowPlanner.plan(melFrames: mel.frames)[0]
        let logits = try await model.logits(mel: mel, start: 0, valid: window.validFrames)
        let frames = WindowPlanner.encoderFrames(window.validFrames)
        let langs = candidates ?? vocabulary.languages
        return try logits.withUnsafeBufferPointer { buffer in
            try LanguageScorer.rank(logits: buffer, frames: frames, classCount: model.classCount,
                                    columns: langs.map { ($0, try vocabulary.columns(for: $0)) }, blank: vocabulary.blank)
        }
    }

    public func detectLanguage(contentsOf url: URL, candidates: [IndicLanguage]? = nil) async throws -> [LanguageScore] {
        try await detectLanguage(audio: AudioLoader.load(url, maxSeconds: 15.4), candidates: candidates)
    }
}

// MARK: - Audio

/// Decodes any file AVFoundation reads (wav, m4a, mov, mp4, …) to 16 kHz mono Float.
public enum AudioLoader {
    public static func load(_ url: URL, maxSeconds: Double? = nil) throws -> [Float] {
        let file: AVAudioFile
        do { file = try AVAudioFile(forReading: url) } catch { throw IndicSpeechError.audio("can't open \(url.lastPathComponent): \(error.localizedDescription)") }
        guard let target = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: IndicConformerEngine.sampleRate, channels: 1, interleaved: false),
              let converter = AVAudioConverter(from: file.processingFormat, to: target) else {
            throw IndicSpeechError.audio("can't convert \(url.lastPathComponent) to 16 kHz mono")
        }
        let sourceRate = file.processingFormat.sampleRate
        let limit = min(maxSeconds.map { AVAudioFramePosition($0 * sourceRate) } ?? file.length, file.length)
        let chunk: AVAudioFrameCount = 65_536
        let ratio = IndicConformerEngine.sampleRate / sourceRate
        guard let input = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: chunk),
              let output = AVAudioPCMBuffer(pcmFormat: target, frameCapacity: AVAudioFrameCount(Double(chunk) * ratio) + 1024) else {
            throw IndicSpeechError.audio("can't allocate audio buffers")
        }
        // The converter pulls input through a @Sendable block; it runs synchronously inside convert(to:), so the
        // box only ever hands over the chunk just read, once.
        final class Feed: @unchecked Sendable { var buffer: AVAudioPCMBuffer?; var ended = false }
        let feed = Feed()
        var out: [Float] = []
        out.reserveCapacity(Int(Double(limit) * ratio) + 1024)
        while true {
            let remaining = limit - file.framePosition
            if remaining > 0 {
                do { try file.read(into: input, frameCount: AVAudioFrameCount(min(AVAudioFramePosition(chunk), remaining))) }
                catch { throw IndicSpeechError.audio("read failed: \(error.localizedDescription)") }
            }
            if remaining <= 0 || input.frameLength == 0 { feed.ended = true } else { feed.buffer = input }
            while true {
                var conversionError: NSError?
                let status = converter.convert(to: output, error: &conversionError) { _, inputStatus in
                    if let buffer = feed.buffer { feed.buffer = nil; inputStatus.pointee = .haveData; return buffer }
                    inputStatus.pointee = feed.ended ? .endOfStream : .noDataNow
                    return nil
                }
                if status == .error { throw IndicSpeechError.audio("conversion failed: \(conversionError?.localizedDescription ?? "unknown")") }
                if let data = output.floatChannelData, output.frameLength > 0 {
                    out.append(contentsOf: UnsafeBufferPointer(start: data[0], count: Int(output.frameLength)))
                }
                if status == .endOfStream { return out }
                if status == .inputRanDry || output.frameLength == 0 { break }
            }
        }
    }
}

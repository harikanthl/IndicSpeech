// SPDX-License-Identifier: Apache-2.0
//
// Parity tests against the NeMo reference, with small fixtures made by scripts/indic/make_fixtures.py (no model
// weights): the Swift mel against the fork's preprocessor on 3 s of Telugu, Tamil and Hindi; greedy CTC decoding
// with word timings on NeMo's own logits for one clip per language; the window plan against the Python twin.
//
// With INDIC_SPEECH_MODEL_DIR (converted model + assets) and INDIC_SPEECH_REF_DIR (export_reference.py output,
// with the clips' wavs in its clips/ folder or INDIC_SPEECH_CLIPS_DIR) set, the end-to-end test also runs every
// reference clip through the real engine and checks it against the Torch run of the same windows.

import Foundation
import Testing
@testable import IndicSpeech

enum Fixture {
    static let directory = Bundle.module.resourceURL!.appending(path: "Fixtures")
    static var assets: URL { directory.appending(path: "assets") }

    struct Word: Decodable { var word: String; var start: Double; var end: Double }
    struct NemoWord: Decodable { var word: String; var start_offset: Int; var end_offset: Int }
    struct Entry: Decodable {
        var clip: String
        var mel_frames: Int
        var mel_samples: Int
        var ctc_frames: Int
        var nemo_ctc_text: String
        var greedy_words: [Word]
        var nemo_ctc_words: [NemoWord]?
    }

    static func entries() throws -> [String: Entry] {
        try JSONDecoder().decode([String: Entry].self, from: Data(contentsOf: directory.appending(path: "fixtures.json")))
    }

    static func floats(_ name: String) throws -> [Float] {
        let data = try Data(contentsOf: directory.appending(path: name))
        return data.withUnsafeBytes { raw in
            (0..<(data.count / 4)).map { Float(bitPattern: UInt32(littleEndian: raw.loadUnaligned(fromByteOffset: $0 * 4, as: UInt32.self))) }
        }
    }

    static func halves(_ name: String) throws -> [Float] {
        let data = try Data(contentsOf: directory.appending(path: name))
        return data.withUnsafeBytes { raw in
            (0..<(data.count / 2)).map { Float(Float16(bitPattern: UInt16(littleEndian: raw.loadUnaligned(fromByteOffset: $0 * 2, as: UInt16.self)))) }
        }
    }
}

func collapseSpaces(_ s: String) -> String { s.split(whereSeparator: \.isWhitespace).joined(separator: " ") }

@Suite struct MelParityTests {
    @Test(arguments: ["te", "ta", "hi"])
    func melMatchesNeMo(language: String) throws {
        let entry = try #require(try Fixture.entries()[language])
        let frontend = try IndicMelFrontend(constants: IndicPreprocessorConstants.load(from: Fixture.assets))
        let audio = try AudioLoader.load(Fixture.directory.appending(path: "mel_\(language).wav"))
        #expect(audio.count == entry.mel_samples)
        let mel = frontend.features(audio)
        let reference = try Fixture.floats("mel_\(language).f32")
        #expect(mel.frames == entry.mel_frames)
        #expect(mel.values.count == reference.count)
        var dot = 0.0, a2 = 0.0, b2 = 0.0, maxAbs: Float = 0
        for (x, y) in zip(mel.values, reference) {
            dot += Double(x) * Double(y); a2 += Double(x) * Double(x); b2 += Double(y) * Double(y)
            maxAbs = max(maxAbs, abs(x - y))
        }
        let cosine = dot / (a2.squareRoot() * b2.squareRoot())
        print("mel parity [\(language)]: \(mel.frames) frames, cosine \(cosine), max abs \(maxAbs)")
        #expect(cosine > 0.99999)
        #expect(maxAbs < 1e-3)
    }

    @Test func reflectPaddingMatchesTorch() {
        // torch.nn.functional.pad(tensor([1., 2., 3., 4.]), (2, 2), mode="reflect") == [3, 2, 1, 2, 3, 4, 3, 2]
        #expect(IndicMelFrontend.reflectPad([1, 2, 3, 4], by: 2) == [3, 2, 1, 2, 3, 4, 3, 2])
    }
}

@Suite struct CTCParityTests {
    @Test(arguments: ["te", "ta", "hi"])
    func greedyDecodeMatchesNeMoOnItsLogits(language: String) throws {
        let entry = try #require(try Fixture.entries()[language])
        let vocabulary = try IndicVocabulary.load(from: Fixture.assets)
        let pieces = try vocabulary.pieces(for: IndicLanguage(language))
        // The fixture holds only the language's 257 columns, so the slice is the identity here.
        let logits = try Fixture.halves("ctc_\(language).f16")
        #expect(logits.count == entry.ctc_frames * 257)
        let labels = logits.withUnsafeBufferPointer {
            CTCGreedyDecoder.labels(logits: $0, frames: 0..<entry.ctc_frames, classCount: 257, columns: Array(0..<257))
        }
        let words = CTCGreedyDecoder.words(CTCGreedyDecoder.collapse(labels, blank: 256), pieces: pieces)

        // Text: identical to NeMo's own CTC transcription.
        #expect(collapseSpaces(words.map(\.text).joined(separator: " ")) == collapseSpaces(entry.nemo_ctc_text))
        // Timings: identical to the frame rule computed in Python on the same logits.
        #expect(words.map(\.text) == entry.greedy_words.map(\.word))
        for (w, r) in zip(words, entry.greedy_words) {
            #expect(abs(w.start - r.start) < 1e-6 && abs(w.end - r.end) < 1e-6, "\(w.text): \(w.start)–\(w.end) vs \(r.start)–\(r.end)")
        }
        #expect(words.allSatisfy { $0.confidence > 0 && $0.confidence <= 1 && $0.end > $0.start })
        // NeMo 1.23's own offsets: same words; starts never later than ours (its token start is the previous
        // token's emission), ends where the next word starts.
        if let nemo = entry.nemo_ctc_words, nemo.count == words.count {
            var startDiffs: [Double] = []
            for (w, n) in zip(words, nemo) {
                #expect(w.text == n.word)
                startDiffs.append(w.start - Double(n.start_offset) * 0.08)
            }
            let mean = startDiffs.reduce(0, +) / Double(max(startDiffs.count, 1))
            print("CTC [\(language)]: \(words.count) words; our start − NeMo 1.23 start: mean \(mean) s, max \(startDiffs.max() ?? 0) s")
            #expect(startDiffs.allSatisfy { $0 >= -1e-9 })
        }
    }

    @Test func collapseMergesRepeatsAndDropsBlanks() {
        let labels: [(label: Int, logProbability: Float)] = [(256, 0), (5, -0.1), (5, -0.1), (256, 0), (5, -0.2), (7, -0.3), (256, 0)]
        let tokens = CTCGreedyDecoder.collapse(labels, blank: 256)
        #expect(tokens.map(\.id) == [5, 5, 7])
        #expect(tokens.map(\.firstFrame) == [1, 4, 5])
        #expect(tokens.map(\.lastFrame) == [2, 4, 5])
    }

    @Test func wordsStartAtWordMarks() {
        let pieces = ["\u{2581}", "\u{2581}na", "mas", "te", "\u{2581}ji"]
        let tokens = [CTCToken(id: 0, firstFrame: 2, lastFrame: 2, logProbability: 0), CTCToken(id: 2, firstFrame: 3, lastFrame: 4, logProbability: 0),
                      CTCToken(id: 4, firstFrame: 9, lastFrame: 9, logProbability: 0), CTCToken(id: 1, firstFrame: 12, lastFrame: 12, logProbability: 0),
                      CTCToken(id: 3, firstFrame: 14, lastFrame: 15, logProbability: 0)]
        let words = CTCGreedyDecoder.words(tokens, pieces: pieces)
        #expect(words.map(\.text) == ["mas", "ji", "nate"])
        #expect(words[0].start == 2 * 0.08 && abs(words[0].end - 5 * 0.08) < 1e-12)
        #expect(abs(words[2].start - 12 * 0.08) < 1e-12 && abs(words[2].end - 16 * 0.08) < 1e-12)
    }
}

@Suite struct WindowStitchTests {
    // As on a native speaker's Telugu recording: window B (from frame 4) emits the same tokens 2 frames later than A.
    let pieces = ["\u{2581}పరు", "గుల", "\u{2581}జీ", "వి"]
    var wordStart: [Bool] { pieces.map { $0.hasPrefix("\u{2581}") } + Array(repeating: false, count: 257 - pieces.count) }
    func t(_ id: Int, _ f: Int, _ l: Int? = nil) -> CTCToken { CTCToken(id: id, firstFrame: f, lastFrame: l ?? f, logProbability: 0) }
    var a: [CTCToken] { [t(0, 1), t(1, 4), t(2, 8), t(3, 9)] }      // window 0, frames 0…9
    var b: [CTCToken] { [t(1, 6), t(2, 10), t(3, 11)] }             // window 1, frames 4…13
    var windows: [EncoderWindow] {
        [EncoderWindow(start: 0, validFrames: 80, keepFrom: 0, keepTo: 5), EncoderWindow(start: 32, validFrames: 80, keepFrom: 5, keepTo: 14)]
    }
    func text(_ tokens: [CTCToken]) -> String { CTCGreedyDecoder.words(tokens, pieces: pieces).map(\.text).joined(separator: " ") }

    @Test func aFixedHandOverFrameDoublesTheTail() {
        // Midpoint (frame 5): A's tokens starting before it, B's from it.
        #expect(text(a.filter { $0.firstFrame < 5 } + b.filter { $0.firstFrame >= 5 }) == "పరుగులగుల జీవి")
    }

    @Test func handOverAtASharedWord() {
        let merged = WindowStitcher.merge([a, b], windows: windows, wordStart: wordStart)
        #expect(text(merged) == "పరుగుల జీవి")
        #expect(merged.map(\.firstFrame) == [1, 4, 10, 11])      // B's own times from the shared word on
    }

    @Test func noSharedWordFallsBackToTheMidpoint() {
        let other = [t(1, 6), t(3, 11)]
        #expect(text(WindowStitcher.merge([a, other], windows: windows, wordStart: wordStart)) == "పరుగులగులవి")
    }
}

@Suite struct WindowPlanTests {
    @Test func shortClipIsOnePaddedWindow() {
        #expect(WindowPlanner.plan(melFrames: 100) == [EncoderWindow(start: 0, validFrames: 100, keepFrom: 0, keepTo: 13)])
        #expect(WindowPlanner.encoderFrames(1536) == 192)
    }

    @Test func matchesThePythonPlan() {
        // indic_common.plan_windows / keep_ranges for 3000 and 9001 mel frames.
        let p = WindowPlanner.plan(melFrames: 3000)
        #expect(p.map(\.start) == [0, 728, 1464] && p.map(\.keepFrom) == [0, 141, 233] && p.map(\.keepTo) == [141, 233, 375])
        let q = WindowPlanner.plan(melFrames: 9001)
        #expect(q.map(\.start) == [0, 1240, 2488, 3736, 4976, 6224, 7472])
        #expect(q.map(\.keepTo) == [173, 329, 485, 640, 796, 952, 1126] && q.last?.validFrames == 1529)
    }

    @Test(arguments: [1537, 2000, 4999, 12_000, 360_001])
    func windowsTileTheClip(frames: Int) {
        let plan = WindowPlanner.plan(melFrames: frames)
        #expect(plan.allSatisfy { $0.validFrames <= 1536 && $0.start % 8 == 0 })
        #expect(plan.last.map { $0.start + $0.validFrames } == frames)
        #expect(plan.first?.keepFrom == 0 && plan.last?.keepTo == WindowPlanner.encoderFrames(frames))
        for (a, b) in zip(plan, plan.dropFirst()) {
            #expect(a.keepTo == b.keepFrom)
            #expect(b.start - a.start <= 1336 + 7)
            #expect(a.keepTo > a.encoderOffset && a.keepTo <= a.encoderOffset + 192 && b.keepFrom >= b.encoderOffset)
        }
    }
}

// SPDX-License-Identifier: Apache-2.0
//
// Greedy CTC decoding over one language's slice of the shared head, with word timings from the frames.
//
// Per encoder frame (80 ms) the label is the argmax over the language's 257 columns; the masked log-softmax that
// NeMo applies doesn't change an argmax, so it is only computed for the label's score. Repeats collapse and the
// blank (index 256 in the slice) drops out. A token starts at the first frame of its run and ends after its last
// frame; a piece starting with "▁" begins a word, so a word runs from its first token's first frame to its last
// token's last frame + 1. CTC peaks early and briefly, so these ends are early; callers that have the audio's
// loudness can tighten them.
//
// NeMo 1.23's own CTC word offsets differ: each token's start is the previous token's emission frame, and a word
// ends where the next word starts (the off-by-one NeMo later fixed in PR #16354). The reference script records
// both; this decoder follows the frames.

import Foundation

/// A recognised word with times in seconds from the start of the audio.
public struct IndicWord: Sendable, Equatable, Hashable, Codable {
    public var text: String
    public var start: Double
    public var end: Double
    /// Mean over the word's emitting frames of the label's probability under the language's masked softmax, 0…1.
    public var confidence: Double
    public init(text: String, start: Double, end: Double, confidence: Double) {
        self.text = text; self.start = start; self.end = end; self.confidence = confidence
    }
}

/// One emitted token after collapse.
public struct CTCToken: Sendable, Equatable {
    public var id: Int
    public var firstFrame: Int
    public var lastFrame: Int
    /// Sum of the run's per-frame log-probabilities, and the run length.
    public var logProbability: Double
}

public enum CTCGreedyDecoder {
    public static let frameSeconds = 0.08

    /// Per-frame (label, log-probability) over `columns` of row-major `[frames × classCount]` logits.
    public static func labels(logits: UnsafeBufferPointer<Float>, frames: Range<Int>, classCount: Int, columns: [Int]) -> [(label: Int, logProbability: Float)] {
        var out: [(Int, Float)] = []
        out.reserveCapacity(frames.count)
        var slice = [Float](repeating: 0, count: columns.count)
        for t in frames {
            let row = t * classCount
            var best = 0
            var bestValue = -Float.infinity
            for (j, c) in columns.enumerated() {
                let v = logits[row + c]
                slice[j] = v
                if v > bestValue { bestValue = v; best = j }
            }
            // log-softmax of the winner: v_best − logsumexp(slice)
            var sum: Float = 0
            for v in slice { sum += expf(v - bestValue) }
            out.append((best, -logf(sum)))
        }
        return out
    }

    public static func collapse(_ labels: [(label: Int, logProbability: Float)], blank: Int) -> [CTCToken] {
        var tokens: [CTCToken] = []
        var previous = blank
        for (t, (label, lp)) in labels.enumerated() {
            if label != blank {
                if label == previous {
                    tokens[tokens.count - 1].lastFrame = t
                    tokens[tokens.count - 1].logProbability += Double(lp)
                } else {
                    tokens.append(CTCToken(id: label, firstFrame: t, lastFrame: t, logProbability: Double(lp)))
                }
            }
            previous = label
        }
        return tokens
    }

    public static func words(_ tokens: [CTCToken], pieces: [String], frameOffset: Int = 0) -> [IndicWord] {
        var out: [IndicWord] = []
        var text = ""
        var first = 0, last = 0
        var logSum = 0.0, frameCount = 0
        var started = false
        func flush() {
            guard !text.isEmpty else { return }
            out.append(IndicWord(text: text, start: Double(first + frameOffset) * frameSeconds,
                                 end: Double(last + 1 + frameOffset) * frameSeconds,
                                 confidence: frameCount > 0 ? exp(logSum / Double(frameCount)) : 0))
        }
        for token in tokens {
            guard token.id < pieces.count else { continue }
            let piece = pieces[token.id]
            let run = token.lastFrame - token.firstFrame + 1
            if piece.hasPrefix("\u{2581}") || !started {
                flush()
                started = true
                text = String(piece.drop(while: { $0 == "\u{2581}" }))
                first = token.firstFrame
                logSum = 0; frameCount = 0
            } else {
                text += piece
            }
            last = token.lastFrame
            logSum += token.logProbability
            frameCount += run
        }
        flush()
        return out
    }
}

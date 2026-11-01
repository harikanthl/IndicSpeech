// SPDX-License-Identifier: Apache-2.0
//
// Spoken-language ID from the CTC head the model already computes. The head is shared: each language is a
// 257-column slice of the same 5,633 logits, and softmax over a slice is that language's distribution. Speech
// in language A scored under language B's slice gives flatter, less confident peaks, so the mean confidence of
// the emitted (non-blank) frames separates languages without a separate classifier and costs one window.
//
// Score per language: the mean, over frames whose argmax in the slice is not the blank, of that argmax's
// log-probability. A language with no emitting frame scores −∞. Measured on FLEURS te/ta/hi in
// docs/research/indic-conformer-apple-silicon-2026-10.md (P0 results).

import Foundation

public struct LanguageScore: Sendable, Equatable {
    public var language: IndicLanguage
    /// Mean log-probability of the emitted labels (≤ 0; closer to 0 is more confident).
    public var score: Double
    /// Share of frames with a non-blank label.
    public var emittingShare: Double
}

public enum LanguageScorer {
    public static func rank(logits: UnsafeBufferPointer<Float>, frames: Int, classCount: Int,
                            columns: [(IndicLanguage, [Int])], blank: Int) -> [LanguageScore] {
        columns.map { language, cols in
            let labels = CTCGreedyDecoder.labels(logits: logits, frames: 0..<frames, classCount: classCount, columns: cols)
            let emitting = labels.filter { $0.label != blank }
            let score = emitting.isEmpty ? -Double.infinity : emitting.reduce(0.0) { $0 + Double($1.logProbability) } / Double(emitting.count)
            return LanguageScore(language: language, score: score, emittingShare: Double(emitting.count) / Double(max(frames, 1)))
        }
        .sorted { $0.score > $1.score }
    }
}

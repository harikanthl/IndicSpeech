// SPDX-License-Identifier: Apache-2.0
//
// How a clip of any length is cut into the encoder's fixed 15.36 s input, and how the windows' outputs are put
// back together. The model is full-context (trained with unlimited attention), so it can't stream; it runs on
// fixed [1, 80, 1536] mel windows, which give 192 encoder frames (three stride-2 convolutions, 80 ms each).
//
// A clip that fits is one window, zero-padded, with its true length passed so the encoder masks the padding.
// A longer clip gets the fewest full windows whose hop is at most 1,336 mel frames (13.36 s, so neighbours
// overlap by 2 s or more; FluidAudio measured shorter overlaps costing accuracy on the same encoder), spread
// evenly, with starts on multiples of 8 mel frames so every window's encoder frames line up with the clip's,
// and the last window ending at the clip's end. Neighbours split their overlap at its midpoint, by encoder
// frame by default; WindowStitcher moves the hand-over to a word both windows emit, so each token comes from
// exactly one window. Each window's mel comes from its own audio, normalised over itself.
//
// scripts/indic/indic_common.py is the same plan in Python; the tests hold the two together.

import Foundation

public struct EncoderWindow: Sendable, Equatable {
    /// First mel frame of the window in the clip.
    public var start: Int
    /// Valid mel frames (≤ windowFrames); the rest is padding.
    public var validFrames: Int
    /// Global encoder frames [keepFrom, keepTo) whose labels this window contributes.
    public var keepFrom: Int
    public var keepTo: Int
    /// Global index of the window's first encoder frame.
    public var encoderOffset: Int { start / WindowPlanner.subsampling }
}

public enum WindowPlanner {
    public static let windowFrames = 1536
    public static let maxHopFrames = 1336
    public static let subsampling = 8

    /// Encoder output length for `melFrames` valid frames.
    public static func encoderFrames(_ melFrames: Int) -> Int {
        var n = melFrames
        for _ in 0..<3 { n = (n + 2 - 3) / 2 + 1 }
        return n
    }

    public static func plan(melFrames: Int) -> [EncoderWindow] {
        guard melFrames > windowFrames else {
            return [EncoderWindow(start: 0, validFrames: melFrames, keepFrom: 0, keepTo: encoderFrames(melFrames))]
        }
        let excess = melFrames - windowFrames
        let last = (excess + subsampling - 1) / subsampling * subsampling
        let gaps = (last + maxHopFrames - 1) / maxHopFrames
        let starts = (0..<gaps).map { $0 * last / gaps / subsampling * subsampling } + [last]
        let spans = starts.map { ($0 / subsampling, $0 / subsampling + encoderFrames(min(windowFrames, melFrames - $0))) }
        let cuts = (0..<(spans.count - 1)).map { (spans[$0 + 1].0 + spans[$0].1) / 2 }
        return starts.indices.map { i in
            EncoderWindow(start: starts[i], validFrames: min(windowFrames, melFrames - starts[i]),
                          keepFrom: i == 0 ? spans[0].0 : cuts[i - 1],
                          keepTo: i == starts.count - 1 ? spans[i].1 : cuts[i])
        }
    }
}

/// Joins the windows' token lists. Each window is greedy-decoded and collapsed on its own (tokens in global
/// frames). Neighbours hand over at a word both of them emit: a word-initial piece with the same id in A and B,
/// inside their overlap, at frames at most `matchFrames` apart, nearest the overlap's midpoint. A keeps its tokens
/// before that word; B supplies the word and what follows, each with its own frame times.
///
/// Why tokens and not frames: CTC emits later when a frame has less future context, so near A's end the two
/// windows place the same tokens a few frames apart (3 frames, 240 ms, on a native speaker's Telugu recording). Any fixed
/// hand-over frame inside a word then keeps that word's tail from both windows: "పరుగుల" came out "పరుగులగుల" and
/// "ఉంచుదాం ధన్యవాదాలు" came out "ఉంచుదాం ం ధన్యవాదాలు". With no shared word in the overlap, A keeps the tokens
/// that start before the midpoint and B the rest.
public enum WindowStitcher {
    public static let matchFrames = 6

    /// `tokens[i]`: window i's collapsed tokens, frames already global.
    public static func merge(_ tokens: [[CTCToken]], windows: [EncoderWindow], wordStart: [Bool]) -> [CTCToken] {
        guard tokens.count == windows.count, var out = tokens.first else { return [] }
        for i in 1..<max(tokens.count, 1) {
            let b = tokens[i]
            let overlap = windows[i].encoderOffset..<(windows[i - 1].encoderOffset + WindowPlanner.encoderFrames(windows[i - 1].validFrames))
            let midpoint = windows[i - 1].keepTo
            var best: (distance: Int, ia: Int, ib: Int)?
            for ia in stride(from: out.count - 1, through: 0, by: -1) {
                let ta = out[ia]
                if ta.firstFrame < overlap.lowerBound { break }
                guard overlap.contains(ta.firstFrame), ta.id < wordStart.count, wordStart[ta.id] else { continue }
                for (ib, tb) in b.enumerated() where tb.id == ta.id && abs(tb.firstFrame - ta.firstFrame) <= matchFrames {
                    let distance = abs(ta.firstFrame - midpoint)
                    if best == nil || distance < best!.distance { best = (distance, ia, ib) }
                }
            }
            if let best {
                out = Array(out[..<best.ia]) + Array(b[best.ib...])
            } else {
                out = out.filter { $0.firstFrame < midpoint } + b.filter { $0.firstFrame >= midpoint }
            }
        }
        return out
    }
}

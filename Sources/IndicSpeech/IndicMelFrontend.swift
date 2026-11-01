// SPDX-License-Identifier: Apache-2.0 AND MIT
//
// IndicMelFrontend turns 16 kHz mono audio into the 80-bin log-mel features IndicConformer was trained on. It
// reproduces the AI4Bharat NeMo fork's AudioToMelSpectrogramPreprocessor in eval mode: 0.97 preemphasis,
// torch.stft(n_fft 512, hop 160, win 400, center: true) with reflect padding and a symmetric Hann window, power
// spectrum, the checkpoint's own Slaney filterbank, log(x + 2^-24), then per-feature normalisation over the
// valid frames with the unbiased standard deviation plus 1e-5. No dither.
//
// The window and filterbank are not recomputed: they are read from the preprocessor constants dumped from the
// `.nemo` (scripts/indic/export_reference.py), so the features match the reference bit for bit up to FFT
// rounding.
//
// Ported from Muesli's retired IndicConformer backend (IndicASRBackend.swift, IndicASRMelSpectrogram, at
// commit 3b04e2ee73, https://github.com/Muesli-HQ/muesli), Copyright (c) 2026 Pranav Hari, MIT License; see
// NOTICE. Changed from the original: the vDSP real FFT's factor of 2 is removed so the power spectrum equals
// torch's; normalisation uses std + 1e-5 exactly as the fork's normalize_batch (Muesli used
// sqrt(max(var, guard)) + guard) and accumulates in Double; any number of frames instead of a fixed 1,024; the
// constants come from a JSON + float32 file pair instead of Muesli's IASRPC01 blob.

import Accelerate
import Foundation

/// The mel preprocessor's fixed numbers and arrays, as dumped from the checkpoint.
public struct IndicPreprocessorConstants: Sendable, Equatable {
    public var sampleRate: Int
    public var nFFT: Int
    public var hopLength: Int
    public var winLength: Int
    public var nMels: Int
    public var preemphasis: Float
    public var logZeroGuard: Float
    public var normalizeGuard: Float
    /// `winLength` values: the Hann window.
    public var window: [Float]
    /// `nMels × (nFFT / 2 + 1)` values, row-major (one row per mel bin).
    public var filterbank: [Float]

    public var bins: Int { nFFT / 2 + 1 }

    /// Reads `preprocessor.json` (the numbers) and `preprocessor.bin` (window then filterbank, little-endian
    /// float32) from `directory`.
    public static func load(from directory: URL) throws -> IndicPreprocessorConstants {
        struct Header: Decodable {
            var sample_rate: Int
            var n_fft: Int
            var hop_length: Int
            var win_length: Int
            var n_mels: Int
            var preemphasis: Float
            var log_zero_guard: Float
            var normalize_guard: Float
        }
        let header = try JSONDecoder().decode(Header.self, from: Data(contentsOf: directory.appending(path: "preprocessor.json")))
        let blob = try Data(contentsOf: directory.appending(path: "preprocessor.bin"))
        let bins = header.n_fft / 2 + 1
        let count = header.win_length + header.n_mels * bins
        guard blob.count == count * MemoryLayout<Float>.size else {
            throw IndicSpeechError.badModelFiles("preprocessor.bin has \(blob.count) bytes, expected \(count * 4)")
        }
        let floats: [Float] = blob.withUnsafeBytes { raw in
            (0..<count).map { Float(bitPattern: UInt32(littleEndian: raw.loadUnaligned(fromByteOffset: $0 * 4, as: UInt32.self))) }
        }
        return IndicPreprocessorConstants(
            sampleRate: header.sample_rate, nFFT: header.n_fft, hopLength: header.hop_length,
            winLength: header.win_length, nMels: header.n_mels, preemphasis: header.preemphasis,
            logZeroGuard: header.log_zero_guard, normalizeGuard: header.normalize_guard,
            window: Array(floats[0..<header.win_length]), filterbank: Array(floats[header.win_length...]))
    }
}

/// Log-mel features, `[nMels][frames]` flattened row-major.
public struct MelFeatures: Sendable {
    public var values: [Float]
    public var mels: Int
    public var frames: Int

    public subscript(mel: Int, frame: Int) -> Float { values[mel * frames + frame] }
}

/// Stateless apart from its FFT setup; safe to share across tasks.
public final class IndicMelFrontend: @unchecked Sendable {
    public let constants: IndicPreprocessorConstants
    private let fft: FFTSetup
    private let log2n: vDSP_Length

    public init(constants: IndicPreprocessorConstants) throws {
        guard constants.nFFT > 0, constants.nFFT & (constants.nFFT - 1) == 0,
              constants.window.count == constants.winLength, constants.winLength <= constants.nFFT,
              constants.filterbank.count == constants.nMels * constants.bins else {
            throw IndicSpeechError.badModelFiles("preprocessor constants have inconsistent shapes")
        }
        self.constants = constants
        log2n = vDSP_Length(constants.nFFT.trailingZeroBitCount)
        guard let setup = vDSP_create_fftsetup(log2n, FFTRadix(kFFTRadix2)) else {
            throw IndicSpeechError.badModelFiles("could not create an FFT of size \(constants.nFFT)")
        }
        fft = setup
    }

    deinit { vDSP_destroy_fftsetup(fft) }

    /// Number of mel frames NeMo produces for `samples` samples (center: true): `samples / hop + 1`.
    public func frameCount(samples: Int) -> Int { samples / constants.hopLength + 1 }

    /// Features for the whole of `audio` (16 kHz mono, any length ≥ 2 samples), normalised over all its frames.
    public func features(_ audio: [Float]) -> MelFeatures {
        let c = constants
        let nFFT = c.nFFT, half = nFFT / 2, bins = c.bins, hop = c.hopLength
        let frames = frameCount(samples: audio.count)
        guard audio.count > 1 else { return MelFeatures(values: [Float](repeating: 0, count: c.nMels * frames), mels: c.nMels, frames: frames) }

        // Preemphasis, then reflect padding of n_fft / 2 on both sides (torch.stft center: true).
        var emphasized = [Float](repeating: 0, count: audio.count)
        emphasized[0] = audio[0]
        for i in 1..<audio.count { emphasized[i] = audio[i] - c.preemphasis * audio[i - 1] }
        let padded = Self.reflectPad(emphasized, by: half)

        // Power spectrum, one row of `bins` per frame. torch.stft pads the 400-sample window to 512 centred.
        let windowOffset = (nFFT - c.winLength) / 2
        var power = [Float](repeating: 0, count: frames * bins)
        var frame = [Float](repeating: 0, count: nFFT)
        var re = [Float](repeating: 0, count: half)
        var im = [Float](repeating: 0, count: half)
        padded.withUnsafeBufferPointer { src in
            for f in 0..<frames {
                frame.withUnsafeMutableBufferPointer { dst in
                    vDSP_vclr(dst.baseAddress!, 1, vDSP_Length(nFFT))
                    vDSP_vmul(src.baseAddress! + f * hop + windowOffset, 1, c.window, 1,
                              dst.baseAddress! + windowOffset, 1, vDSP_Length(c.winLength))
                }
                re.withUnsafeMutableBufferPointer { reBuf in
                    im.withUnsafeMutableBufferPointer { imBuf in
                        var split = DSPSplitComplex(realp: reBuf.baseAddress!, imagp: imBuf.baseAddress!)
                        frame.withUnsafeBufferPointer { fb in
                            fb.baseAddress!.withMemoryRebound(to: DSPComplex.self, capacity: half) {
                                vDSP_ctoz($0, 2, &split, 1, vDSP_Length(half))
                            }
                        }
                        vDSP_fft_zrip(fft, &split, 1, log2n, FFTDirection(kFFTDirection_Forward))
                        // zrip returns 2 × the DFT: packed DC in realp[0], Nyquist in imagp[0].
                        let row = f * bins
                        power[row] = 0.25 * reBuf[0] * reBuf[0]
                        power[row + half] = 0.25 * imBuf[0] * imBuf[0]
                        for k in 1..<half {
                            power[row + k] = 0.25 * (reBuf[k] * reBuf[k] + imBuf[k] * imBuf[k])
                        }
                    }
                }
            }
        }

        // mel[nMels × frames] = filterbank[nMels × bins] · powerᵀ[bins × frames].
        var mel = [Float](repeating: 0, count: c.nMels * frames)
        vDSP_mmul(c.filterbank, 1, Self.transpose(power, rows: frames, columns: bins), 1, &mel, 1,
                  vDSP_Length(c.nMels), vDSP_Length(frames), vDSP_Length(bins))
        var guardValue = c.logZeroGuard
        vDSP_vsadd(mel, 1, &guardValue, &mel, 1, vDSP_Length(mel.count))
        var n = Int32(mel.count)
        vvlogf(&mel, mel, &n)

        // Per-feature normalisation: (x − mean) / (std_unbiased + 1e-5) over every frame.
        for m in 0..<c.nMels {
            let row = m * frames
            var sum = 0.0
            for f in 0..<frames { sum += Double(mel[row + f]) }
            let mean = sum / Double(frames)
            var squares = 0.0
            for f in 0..<frames { let d = Double(mel[row + f]) - mean; squares += d * d }
            let std = (squares / Double(max(frames - 1, 1))).squareRoot() + Double(c.normalizeGuard)
            for f in 0..<frames { mel[row + f] = Float((Double(mel[row + f]) - mean) / std) }
        }
        return MelFeatures(values: mel, mels: c.nMels, frames: frames)
    }

    private static func transpose(_ m: [Float], rows: Int, columns: Int) -> [Float] {
        var out = [Float](repeating: 0, count: m.count)
        vDSP_mtrans(m, 1, &out, 1, vDSP_Length(columns), vDSP_Length(rows))
        return out
    }

    /// numpy/torch "reflect" padding (the edge sample is not repeated).
    static func reflectPad(_ x: [Float], by pad: Int) -> [Float] {
        let n = x.count
        func reflect(_ i: Int) -> Int {
            let period = 2 * n - 2
            var j = i % period
            if j < 0 { j += period }
            return j >= n ? period - j : j
        }
        var out = [Float](repeating: 0, count: n + 2 * pad)
        for i in 0..<out.count { out[i] = x[reflect(i - pad)] }
        return out
    }
}

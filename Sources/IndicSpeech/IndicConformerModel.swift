// SPDX-License-Identifier: Apache-2.0
//
// The Core ML encoder + shared CTC head: one call per 15.36 s window, mel [1, 80, 1536] and its valid length in,
// raw logits [1, 192, 5633] out (no language mask, no log-softmax; see scripts/indic/convert_coreml.py).
//
// A `.mlpackage` is compiled once into a `.mlmodelc` beside it (or in `compiledDirectory`) and the compiled model
// is reused while it is newer than the package, so only the first load pays the compile. Compute units are
// chosen at load: the Neural Engine by default, which leaves the GPU to video work.

import CoreML
import Foundation

public final class IndicConformerModel: @unchecked Sendable {
    public static let melBins = 80

    public let url: URL
    public let windowFrames: Int
    public let encoderFrames: Int
    public let classCount: Int
    /// Seconds spent compiling (0 when a compiled model was reused) and loading.
    public let compileSeconds: Double
    public let loadSeconds: Double
    private let model: MLModel

    /// - Parameters:
    ///   - url: an `.mlpackage` or `.mlmodelc`.
    ///   - compiledDirectory: where to keep the compiled model for an `.mlpackage`; its own folder by default.
    public init(url: URL, computeUnits: MLComputeUnits = .cpuAndNeuralEngine, compiledDirectory: URL? = nil) async throws {
        self.url = url
        let fm = FileManager.default
        guard fm.fileExists(atPath: url.path) else { throw IndicSpeechError.badModelFiles("no model at \(url.path)") }
        var compiled = url
        var compileSeconds = 0.0
        if url.pathExtension == "mlpackage" {
            let dir = compiledDirectory ?? url.deletingLastPathComponent()
            compiled = dir.appending(path: url.deletingPathExtension().lastPathComponent + ".mlmodelc")
            if !Self.isFresh(compiled: compiled, source: url) {
                let clock = ContinuousClock.now
                let temporary = try await MLModel.compileModel(at: url)
                try? fm.removeItem(at: compiled)
                try fm.createDirectory(at: dir, withIntermediateDirectories: true)
                try fm.moveItem(at: temporary, to: compiled)
                compileSeconds = (ContinuousClock.now - clock).seconds
            }
        }
        let configuration = MLModelConfiguration()
        configuration.computeUnits = computeUnits
        let clock = ContinuousClock.now
        model = try await MLModel.load(contentsOf: compiled, configuration: configuration)
        loadSeconds = (ContinuousClock.now - clock).seconds
        self.compileSeconds = compileSeconds

        let inputs = model.modelDescription.inputDescriptionsByName
        let outputs = model.modelDescription.outputDescriptionsByName
        guard let mel = inputs["mel"]?.multiArrayConstraint?.shape.map(\.intValue), mel.count == 3, mel[1] == Self.melBins,
              inputs["length"] != nil,
              let logits = outputs["logits"]?.multiArrayConstraint?.shape.map(\.intValue), logits.count == 3 else {
            throw IndicSpeechError.badModelFiles("\(url.lastPathComponent) doesn't have mel/length inputs and a logits output")
        }
        windowFrames = mel[2]
        encoderFrames = logits[1]
        classCount = logits[2]
    }

    private static func isFresh(compiled: URL, source: URL) -> Bool {
        let fm = FileManager.default
        guard let c = try? fm.attributesOfItem(atPath: compiled.path)[.modificationDate] as? Date,
              let s = try? fm.attributesOfItem(atPath: source.appending(path: "Manifest.json").path)[.modificationDate] as? Date
        else { return false }
        return c >= s
    }

    /// Raw logits `[encoderFrames × classCount]`, row-major, for one window of `mel` starting at mel frame `start`
    /// with `valid` frames (the rest is zero-padded).
    public func logits(mel: MelFeatures, start: Int, valid: Int) async throws -> [Float] {
        let input = try MLMultiArray(shape: [1, NSNumber(value: Self.melBins), NSNumber(value: windowFrames)], dataType: .float32)
        input.withUnsafeMutableBytes { raw, strides in
            let p = raw.bindMemory(to: Float.self)
            let rowStride = strides[1], colStride = strides[2]
            for m in 0..<Self.melBins {
                for f in 0..<windowFrames {
                    p[m * rowStride + f * colStride] = f < valid ? mel.values[m * mel.frames + start + f] : 0
                }
            }
        }
        let length = try MLMultiArray(shape: [1], dataType: .int32)
        length[0] = NSNumber(value: Int32(valid))
        let features = try MLDictionaryFeatureProvider(dictionary: ["mel": MLFeatureValue(multiArray: input),
                                                                    "length": MLFeatureValue(multiArray: length)])
        let result = try await model.prediction(from: features)
        guard let out = result.featureValue(for: "logits")?.multiArrayValue else {
            throw IndicSpeechError.model("prediction returned no logits")
        }
        return Self.contiguous(out, rows: encoderFrames, columns: classCount)
    }

    /// Copies a [1, rows, columns] array into row-major Floats, whatever its strides and element type.
    static func contiguous(_ array: MLMultiArray, rows: Int, columns: Int) -> [Float] {
        var out = [Float](repeating: 0, count: rows * columns)
        let strides = array.strides.map(\.intValue)
        array.withUnsafeBytes { raw in
            switch array.dataType {
            case .float16:
                let p = raw.bindMemory(to: Float16.self)
                for r in 0..<rows { for c in 0..<columns { out[r * columns + c] = Float(p[r * strides[1] + c * strides[2]]) } }
            default:
                let p = raw.bindMemory(to: Float.self)
                if strides[2] == 1, strides[1] == columns {
                    out.withUnsafeMutableBufferPointer { $0.baseAddress!.update(from: p.baseAddress!, count: rows * columns) }
                } else {
                    for r in 0..<rows { for c in 0..<columns { out[r * columns + c] = p[r * strides[1] + c * strides[2]] } }
                }
            }
        }
        return out
    }
}

extension Duration {
    var seconds: Double { Double(components.seconds) + Double(components.attoseconds) * 1e-18 }
}

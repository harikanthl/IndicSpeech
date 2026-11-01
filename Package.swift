// swift-tools-version: 6.2
// SPDX-License-Identifier: Apache-2.0
import PackageDescription

// On-device speech recognition for Indian languages with AI4Bharat's IndicConformer 600M (multilingual,
// CTC head) converted to Core ML. Standalone: system frameworks only (Accelerate, AVFoundation, Core ML).
// The model itself is downloaded separately; see README.md.
let package = Package(
    name: "IndicSpeech",
    platforms: [.macOS(.v26), .iOS(.v26)],
    products: [
        .library(name: "IndicSpeech", targets: ["IndicSpeech"]),
        .executable(name: "indic-transcribe", targets: ["indic-transcribe"]),
    ],
    targets: [
        .target(name: "IndicSpeech"),
        .executableTarget(name: "indic-transcribe", dependencies: ["IndicSpeech"]),
        .testTarget(name: "IndicSpeechTests", dependencies: ["IndicSpeech"], resources: [.copy("Fixtures")]),
    ],
    swiftLanguageModes: [.v6]
)

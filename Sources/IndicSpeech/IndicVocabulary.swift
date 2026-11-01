// SPDX-License-Identifier: Apache-2.0
//
// The model's languages, their SentencePiece pieces and their columns in the shared CTC head. IndicConformer
// 600M multilingual has one CTC projection with 5,633 outputs: 22 languages × 256 pieces plus one blank. A
// language is a 257-column slice (its 256 pieces, then the blank), read from language_columns.json, which
// scripts/indic/export_reference.py dumps from the checkpoint's language masks. vocab.json holds each language's
// pieces in token-id order; a piece that starts with "▁" (U+2581) begins a word.

import Foundation

/// A language the model knows, by its ISO 639 code as the checkpoint names it ("te", "ta", "hi", …).
public struct IndicLanguage: RawRepresentable, Hashable, Sendable, Codable, CustomStringConvertible {
    public var rawValue: String
    public init(rawValue: String) { self.rawValue = rawValue }
    public init(_ code: String) { rawValue = code }
    public var description: String { rawValue }

    public static let assamese = Self("as"), bengali = Self("bn"), bodo = Self("brx"), dogri = Self("doi")
    public static let gujarati = Self("gu"), hindi = Self("hi"), kannada = Self("kn"), konkani = Self("kok")
    public static let kashmiri = Self("ks"), maithili = Self("mai"), malayalam = Self("ml"), manipuri = Self("mni")
    public static let marathi = Self("mr"), nepali = Self("ne"), odia = Self("or"), punjabi = Self("pa")
    public static let sanskrit = Self("sa"), santali = Self("sat"), sindhi = Self("sd"), tamil = Self("ta")
    public static let telugu = Self("te"), urdu = Self("ur")
}

public struct IndicVocabulary: Sendable {
    /// Languages in the checkpoint's order.
    public let languages: [IndicLanguage]
    /// Width of the CTC head (5,633).
    public let classCount: Int
    /// Index of the blank within a language's slice (256).
    public let blank: Int
    private let pieces: [IndicLanguage: [String]]
    private let columns: [IndicLanguage: [Int]]

    public init(pieces: [IndicLanguage: [String]], columns: [IndicLanguage: [Int]], order: [IndicLanguage], classCount: Int) throws {
        for language in order {
            guard let p = pieces[language], let c = columns[language], c.count == p.count + 1,
                  c.allSatisfy({ (0..<classCount).contains($0) }) else {
                throw IndicSpeechError.badModelFiles("vocabulary and columns disagree for \(language)")
            }
        }
        self.pieces = pieces
        self.columns = columns
        languages = order
        self.classCount = classCount
        blank = (columns[order[0]]?.count ?? 1) - 1
    }

    /// Reads vocab.json, language_columns.json and assets.json from `directory`.
    public static func load(from directory: URL) throws -> IndicVocabulary {
        struct Info: Decodable { var languages: [String]; var num_classes: Int }
        let info = try JSONDecoder().decode(Info.self, from: Data(contentsOf: directory.appending(path: "assets.json")))
        let vocab = try JSONDecoder().decode([String: [String]].self, from: Data(contentsOf: directory.appending(path: "vocab.json")))
        let cols = try JSONDecoder().decode([String: [Int]].self, from: Data(contentsOf: directory.appending(path: "language_columns.json")))
        let order = info.languages.map(IndicLanguage.init(rawValue:))
        return try IndicVocabulary(
            pieces: Dictionary(uniqueKeysWithValues: vocab.map { (IndicLanguage($0.key), $0.value) }),
            columns: Dictionary(uniqueKeysWithValues: cols.map { (IndicLanguage($0.key), $0.value) }),
            order: order, classCount: info.num_classes)
    }

    public func supports(_ language: IndicLanguage) -> Bool { columns[language] != nil }

    /// The language's 257 columns in the 5,633-wide head; the last is the blank.
    public func columns(for language: IndicLanguage) throws -> [Int] {
        guard let c = columns[language] else { throw IndicSpeechError.unsupportedLanguage(language) }
        return c
    }

    public func pieces(for language: IndicLanguage) throws -> [String] {
        guard let p = pieces[language] else { throw IndicSpeechError.unsupportedLanguage(language) }
        return p
    }
}

public enum IndicSpeechError: Error, Sendable, Equatable, CustomStringConvertible {
    case badModelFiles(String)
    case unsupportedLanguage(IndicLanguage)
    case audio(String)
    case model(String)

    public var description: String {
        switch self {
        case .badModelFiles(let why): "IndicSpeech model files: \(why)"
        case .unsupportedLanguage(let language): "IndicSpeech: the model has no language \"\(language)\""
        case .audio(let why): "IndicSpeech audio: \(why)"
        case .model(let why): "IndicSpeech model: \(why)"
        }
    }
}

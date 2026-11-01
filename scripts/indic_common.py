# SPDX-License-Identifier: Apache-2.0
"""Shared, dependency-light helpers (numpy only) for the IndicConformer CTC pipeline.

These are the Python twins of the Swift package's WindowPlanner and CTCGreedyDecoder, so the reference, the bench
and the app all window, stitch and decode the same way:

- Windows (each featurised from its own audio and normalised over itself): the encoder runs on a fixed [1, 80, 1536] mel input (15.36 s, 192 encoder frames at 80 ms). A clip that
  fits is one zero-padded window with its true length. A longer clip is cut into full 1536-frame windows whose
  starts are multiples of 8 mel frames (one encoder frame), the fewest windows whose hop is at most 1336 frames
  (so neighbours overlap by about 2 s or more), spread evenly with the last one ending at the clip's end. Each pair of neighbours splits its overlap at the midpoint, by encoder
  frame; each window is collapsed on its own and the token lists are joined at a word both windows emit
  (stitch_tokens).
- Greedy CTC over one language's 257 columns (blank = 256); a token starts at the first frame of its run and ends
  after the last; a SentencePiece piece starting with U+2581 begins a word.
"""
from __future__ import annotations

import unicodedata

import numpy as np

WINDOW_MEL_FRAMES = 1536
HOP_MEL_FRAMES = 1336
SUBSAMPLING = 8
FRAME_SECONDS = 0.08
BLANK = 256
WORD_MARK = "▁"


def encoder_frames(mel_frames: int) -> int:
    """Encoder output length for `mel_frames` valid inputs: three stride-2, kernel-3, pad-1 convolutions."""
    n = mel_frames
    for _ in range(3):
        n = (n + 2 * 1 - 3) // 2 + 1
    return n


def plan_windows(mel_frames: int) -> list[tuple[int, int]]:
    """[(start, valid_frames)] in mel frames, covering the clip."""
    if mel_frames <= WINDOW_MEL_FRAMES:
        return [(0, mel_frames)]
    last = -(-(mel_frames - WINDOW_MEL_FRAMES) // SUBSAMPLING) * SUBSAMPLING  # round up: last window ≤ 1536
    gaps = -(-last // HOP_MEL_FRAMES)  # fewest windows with a hop ≤ 1336 frames, then spread evenly
    starts = [i * last // gaps // SUBSAMPLING * SUBSAMPLING for i in range(gaps)] + [last]
    return [(s, min(WINDOW_MEL_FRAMES, mel_frames - s)) for s in starts]


HOP_SAMPLES = 160


def window_samples(start: int, valid: int) -> tuple[int, int]:
    """The audio [from, to) whose centred mel frames are exactly the window's frames start … start + valid − 1.
    Each window is featurised as its own clip, normalised over itself, as NeMo's buffered inference does:
    normalising once over a long clip and slicing left one Tamil window entirely blank (2026-11-01)."""
    return start * HOP_SAMPLES, (start + valid - 1) * HOP_SAMPLES + 1


def window_input(mel: np.ndarray, start: int = 0, valid: int | None = None) -> np.ndarray:
    """[80, frames] mel → [1, 80, 1536] zero-padded model input (frames start … start + valid)."""
    valid = mel.shape[1] - start if valid is None else valid
    out = np.zeros((1, mel.shape[0], WINDOW_MEL_FRAMES), dtype=np.float32)
    out[0, :, :valid] = mel[:, start:start + valid]
    return out


def keep_ranges(windows: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """For each window, the global encoder frames [from, to) whose labels it contributes."""
    spans = [(s // SUBSAMPLING, s // SUBSAMPLING + encoder_frames(v)) for s, v in windows]
    cuts = [(spans[i + 1][0] + spans[i][1]) // 2 for i in range(len(spans) - 1)]
    lows = [spans[0][0]] + cuts
    highs = cuts + [spans[-1][1]]
    return list(zip(lows, highs))


def window_labels(logits: np.ndarray, valid: int, columns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One window's [192, 5633] logits → (argmax labels in the 257 space, their log-prob) over its valid frames."""
    sub = logits[: encoder_frames(valid)][:, columns].astype(np.float32)
    lp = sub - np.logaddexp.reduce(sub, axis=1, keepdims=True)
    return sub.argmax(axis=1), lp.max(axis=1)


MATCH_FRAMES = 6


def stitch_tokens(window_logits: list[np.ndarray], windows: list[tuple[int, int]], columns: np.ndarray,
                  word_start: np.ndarray | None = None) -> list[tuple[int, int, int]]:
    """Per-window [192, 5633] logits → one token list [(token, first_frame, last_frame)] in global frames.

    Each window is greedy-decoded and collapsed on its own. Neighbours hand over at a word both of them emit:
    a word-initial piece with the same id in A and B, inside their overlap, at frames no more than MATCH_FRAMES
    apart, nearest the overlap's midpoint. A keeps its tokens before that word, B supplies the word and what
    follows, each with its own frame times. CTC emits later when a frame has less future context, so near A's
    end the two windows disagree by a few frames, and a hand-over at a fixed frame doubled a word's tail
    ("పరుగుల" → "పరుగులగుల", "ఉంచుదాం ధన్యవాదాలు" → "ఉంచుదాం ం ధన్యవాదాలు" on a native speaker's Telugu recording). With no shared
    word in the overlap, or no `word_start`, A keeps tokens starting before the midpoint and B the rest."""
    per = []
    for logits, (s, v) in zip(window_logits, windows):
        labels, _ = window_labels(logits, v, columns)
        base = s // SUBSAMPLING
        per.append([(t, f + base, l + base) for t, f, l in collapse(labels)])
    mids = [hi for _, hi in keep_ranges(windows)[:-1]]
    out = list(per[0])
    for i in range(1, len(per)):
        a, b = out, per[i]
        lo = windows[i][0] // SUBSAMPLING
        hi = windows[i - 1][0] // SUBSAMPLING + encoder_frames(windows[i - 1][1])
        mid = mids[i - 1]
        best = None
        if word_start is not None:
            for ia in range(len(a) - 1, -1, -1):
                ta, fa, _ = a[ia]
                if fa < lo:
                    break
                if fa >= hi or not word_start[ta]:
                    continue
                for ib, (tb, fb, _) in enumerate(b):
                    if tb == ta and abs(fb - fa) <= MATCH_FRAMES and (best is None or abs(fa - mid) < best[0]):
                        best = (abs(fa - mid), ia, ib)
        if best is None:
            out = [x for x in a if x[1] < mid] + [x for x in b if x[1] >= mid]
        else:
            out = a[:best[1]] + b[best[2]:]
    return out


def stitch_labels(window_logits: list[np.ndarray], windows: list[tuple[int, int]], columns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame labels stitched at the overlap midpoints (kept for comparison; stitch_tokens is what the app does)."""
    per = [window_labels(l, v, columns) for l, (_, v) in zip(window_logits, windows)]
    starts = [s // SUBSAMPLING for s, _ in windows]
    ranges = keep_ranges(windows)
    labels = np.concatenate([p[0][lo - s:hi - s] for p, s, (lo, hi) in zip(per, starts, ranges)])
    scores = np.concatenate([p[1][lo - s:hi - s] for p, s, (lo, hi) in zip(per, starts, ranges)])
    return labels, scores


def word_starts(vocab: list[str]) -> np.ndarray:
    """[257] bool over a language's pieces plus the blank."""
    return np.array([p.startswith(WORD_MARK) for p in vocab] + [False])


def collapse(labels: np.ndarray) -> list[tuple[int, int, int]]:
    """[(token, first_frame, last_frame)] after CTC collapse (repeats merged, blanks dropped)."""
    tokens, prev = [], BLANK
    for t, p in enumerate(labels.tolist()):
        if p != BLANK and p != prev:
            tokens.append([p, t, t])
        elif p != BLANK and p == prev:
            tokens[-1][2] = t
        prev = p
    return [tuple(x) for x in tokens]


def words(tokens: list[tuple[int, int, int]], vocab: list[str]) -> list[dict]:
    """Words with start/end in seconds: start = first frame of the first token, end = after the last token's run."""
    out: list[dict] = []
    for tok, first, last in tokens:
        piece = vocab[tok]
        if piece.startswith(WORD_MARK) or not out:
            out.append({"word": piece.lstrip(WORD_MARK), "first": first, "last": last})
        else:
            out[-1]["word"] += piece
            out[-1]["last"] = last
    for w in out:
        w["start"] = round(w["first"] * FRAME_SECONDS, 3)
        w["end"] = round((w["last"] + 1) * FRAME_SECONDS, 3)
    return [w for w in out if w["word"]]


def nemo_rule_words(labels: np.ndarray, vocab: list[str]) -> list[dict]:
    """NeMo 1.23's CTC word offsets, re-implemented (its own timestamp path fails on the multilingual tokenizer):
    a token runs from the previous token's emission frame (the first from its own frame − 1) to its own emission
    frame, and a word ends where the next word starts (ctc_decoding._compute_offsets and
    _get_word_offsets_subwords_sentencepiece)."""
    emits = [(tok, first) for tok, first, _ in collapse(labels)]
    if not emits:
        return []
    starts = [max(0, emits[0][1] - 1)] + [f for _, f in emits[:-1]]
    ends = [f for _, f in emits]
    groups: list[list[int]] = []
    for i, (tok, _) in enumerate(emits):
        if vocab[tok].startswith(WORD_MARK) or not groups:
            groups.append([i])
        else:
            groups[-1].append(i)
    out = []
    for g, nxt in zip(groups, groups[1:] + [None]):
        word = "".join(vocab[emits[i][0]] for i in g).lstrip(WORD_MARK)
        end = starts[nxt[0]] if nxt else ends[g[-1]]
        out.append({"word": word, "start": round(starts[g[0]] * FRAME_SECONDS, 3), "end": round(end * FRAME_SECONDS, 3)})
    return [w for w in out if w["word"]]


def text_of(ws: list[dict]) -> str:
    return " ".join(w["word"] for w in ws)


# --- error rates -------------------------------------------------------------------------------------------------

def normalise(text: str) -> str:
    """NFC, lower case, punctuation (Unicode P*) removed, whitespace collapsed. Danda and double danda are P*."""
    text = unicodedata.normalize("NFC", text).lower()
    text = "".join(" " if unicodedata.category(c).startswith("P") else c for c in text)
    return " ".join(text.split())


def _edits(ref: list, hyp: list) -> int:
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1]


def error_counts(ref: str, hyp: str) -> tuple[int, int, int, int]:
    """(word edits, reference words, char edits, reference chars) after normalise()."""
    r, h = normalise(ref), normalise(hyp)
    return _edits(r.split(), h.split()), len(r.split()), _edits(list(r.replace(" ", "")), list(h.replace(" ", ""))), len(r.replace(" ", ""))


def skeleton(text: str) -> str:
    """normalise(), then drop combining marks (Unicode Mn/Mc: vowel signs, virama, nukta, anusvara…), leaving the
    consonant and independent-vowel skeleton. A CER on this ignores matra and nukta slips."""
    return "".join(c for c in normalise(text) if not unicodedata.category(c).startswith("M"))


def skeleton_cer(ref: str, hyp: str) -> float:
    r, h = skeleton(ref).replace(" ", ""), skeleton(hyp).replace(" ", "")
    return 100 * _edits(list(r), list(h)) / max(len(r), 1)

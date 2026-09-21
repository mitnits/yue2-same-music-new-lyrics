#!/usr/bin/env python3
"""Time the ORIGINAL lyrics against the recording: when is each lyric line sung? Runs in runtime/.venv-describe.

    runtime/.venv-describe/bin/python workers/lyrics_timing_worker.py audio.flac lyrics.txt --language en --output timing.json

Method: Whisper (large-v3-turbo) transcribes the recording with word timestamps; the known lyric words are then
matched to the recognised words in order (sequence alignment on normalised text), and each lyric line gets the time
span of its matched words. Lines with no confident match are interpolated between their neighbours.
Prints one JSON line on stdout: {"status": "ok", "lines": [{index, text, start, end, matched, words}], ...}
"""
from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import sys
import time
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNTIME = Path(os.environ.get("YUE2_RUNTIME", HERE.parent / "runtime")).expanduser()
MODEL_DIR = RUNTIME / "models" / "whisper-large-v3-turbo"

LANG_CODES = {"english": "en", "chinese": "zh", "japanese": "ja", "russian": "ru", "korean": "ko", "spanish": "es",
              "portuguese": "pt", "french": "fr", "german": "de", "polish": "pl", "swedish": "sv", "dutch": "nl",
              "indonesian": "id", "tagalog": "tl", "persian": "fa", "bengali": "bn", "arabic": "ar", "italian": "it",
              "ukrainian": "uk", "hindi": "hi", "urdu": "ur", "hebrew": "he"}


def emit(**payload):
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def norm(word: str) -> str:
    w = unicodedata.normalize("NFKC", word).lower()
    w = re.sub(r"[^\w]", "", w, flags=re.U)
    return w.replace("_", "")


def lyric_lines(text: str) -> list:
    out = []
    for raw in text.splitlines():
        ln = raw.strip()
        if not ln or re.fullmatch(r"\[[^\]]+\]", ln) or re.fullmatch(r"\(.*\)", ln):
            continue
        words = [norm(w) for w in re.split(r"[\s\-–—/]+", ln)]
        words = [w for w in words if w]
        if words:
            out.append({"text": ln, "words": words})
    return out


def transcribe(audio_path: Path, language: str | None):
    import torch
    import librosa
    from transformers import pipeline
    device = "cuda" if torch.cuda.is_available() else "cpu"
    asr = pipeline("automatic-speech-recognition", model=str(MODEL_DIR), device=device,
                   dtype=torch.float16 if device == "cuda" else torch.float32)
    y, sr = librosa.load(str(audio_path), sr=16000, mono=True)
    kwargs = {"task": "transcribe"}
    if language:
        kwargs["language"] = language
    out = asr({"raw": y, "sampling_rate": sr}, return_timestamps="word", chunk_length_s=30, stride_length_s=5,
              generate_kwargs=kwargs)
    words = []
    for ch in out.get("chunks", []):
        t0, t1 = ch.get("timestamp", (None, None))
        if t0 is None:
            continue
        for w in re.split(r"\s+", ch["text"].strip()):
            n = norm(w)
            if n:
                start, end = float(t0), float(t1 if t1 is not None else t0 + 0.3)
                if end - start > 2.0:  # Whisper sometimes stamps a chunk's first word from the chunk start
                    start = end - 0.4
                words.append({"text": w, "norm": n, "start": start, "end": end})
    del asr
    torch.cuda.empty_cache()
    return words, out.get("text", "")


def match_lines(lines: list, words: list) -> list:
    """Sequence-align lyric words to recognised words; derive per-line spans."""
    lyric_seq = [w for L in lines for w in L["words"]]
    heard_seq = [w["norm"] for w in words]
    sm = difflib.SequenceMatcher(a=lyric_seq, b=heard_seq, autojunk=False)
    hit = {}  # lyric word index -> heard word index
    for blk in sm.get_matching_blocks():
        for k in range(blk.size):
            hit[blk.a + k] = blk.b + k
    # fuzzy pass for still-unmatched lyric words: nearest heard word with high similarity inside the neighbourhood
    result, idx = [], 0
    for li, L in enumerate(lines):
        n = len(L["words"])
        matched = [hit[i] for i in range(idx, idx + n) if i in hit]
        entry = {"index": li, "text": L["text"], "matched": len(matched), "total": n}
        if matched:
            entry["start"] = words[min(matched)]["start"]
            entry["end"] = words[max(matched)]["end"]
        result.append(entry)
        idx += n
    # interpolate unmatched lines between neighbours
    known = [i for i, e in enumerate(result) if "start" in e]
    for i, e in enumerate(result):
        if "start" in e:
            continue
        prev = max([k for k in known if k < i], default=None)
        nxt = min([k for k in known if k > i], default=None)
        if prev is not None and nxt is not None:
            span = result[nxt]["start"] - result[prev]["end"]
            gap_lines = nxt - prev
            e["start"] = result[prev]["end"] + span * (i - prev - 1) / gap_lines
            e["end"] = result[prev]["end"] + span * (i - prev) / gap_lines
        elif prev is not None:
            e["start"] = result[prev]["end"]; e["end"] = e["start"] + 3.0
        elif nxt is not None:
            e["end"] = result[nxt]["start"]; e["start"] = max(0.0, e["end"] - 3.0)
        else:
            e["start"] = e["end"] = 0.0
        e["interpolated"] = True
    # enforce monotonic order
    t = 0.0
    for e in result:
        e["start"] = max(e["start"], t); e["end"] = max(e["end"], e["start"] + 0.2); t = e["start"] + 0.2
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path)
    parser.add_argument("lyrics", type=Path)
    parser.add_argument("--language", default="", help="language name or code; empty = let Whisper detect")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    t0 = time.perf_counter()
    try:
        lang = args.language.strip().lower()
        lang = LANG_CODES.get(lang, lang) or None
        lines = lyric_lines(args.lyrics.read_text(encoding="utf-8"))
        if not lines:
            raise ValueError("no lyric lines")
        words, text = transcribe(args.audio, lang)
        result = match_lines(lines, words)
        matched = sum(e["matched"] for e in result); total = sum(e["total"] for e in result)
        payload = {"status": "ok", "language": lang, "lines": result, "heard_words": len(words),
                   "matched_words": matched, "total_words": total, "match_ratio": round(matched / max(1, total), 3),
                   "transcript": text[:2000], "seconds": round(time.perf_counter() - t0, 1)}
    except Exception as exc:
        import traceback
        traceback.print_exc()
        payload = {"status": "error", "error": str(exc)[:500], "seconds": round(time.perf_counter() - t0, 1)}
    if args.output:
        args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    emit(**payload)
    return 0 if payload["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())

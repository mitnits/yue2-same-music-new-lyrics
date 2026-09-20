#!/usr/bin/env python3
"""Transcribe a recording with SheetSage2 into YuE2-native ABC. Runs in .venv-sheetsage2.

    YuE/.venv-sheetsage2/bin/python sheetsage_worker.py song.mp3 --output out_dir [--melody-only] [--device auto|cuda|cpu]

Prints one JSON line on stdout with the outcome. Falls back to CPU on CUDA out-of-memory.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE / "YuE" / "models" / "SheetSage2"


def emit(**payload):
    print(json.dumps(payload), flush=True)


def load(device, dtype):
    import torch
    from transformers import AutoModel
    model = AutoModel.from_pretrained(str(MODEL_DIR), trust_remote_code=True, local_files_only=True).eval()
    return model.to(device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--melody-only", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default=None)
    parser.add_argument("--max-seconds", type=float)
    parser.add_argument("--tasks", choices=("default", "melody-vocal"), default="default",
                        help="default = vocal + instrumental melodies (melody_full); melody-vocal = ask only for the "
                             "sung melody (melody_vocal), useful when the voice gets filed as an instrument")
    args = parser.parse_args()

    import torch
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = args.dtype or ("bf16" if device == "cuda" else "fp32")
    args.output.mkdir(parents=True, exist_ok=True)
    kwargs = dict(output_dir=str(args.output), dtype=dtype, melody_only=args.melody_only)
    if args.tasks == "melody-vocal":
        kwargs["prompts"] = ("timestamp", "downbeat_meter", "structure", "key", "chord_full", "melody_vocal")
    if args.max_seconds:
        kwargs["max_seconds"] = args.max_seconds

    start = time.perf_counter()
    for attempt in ("primary", "cpu-fallback"):
        try:
            t0 = time.perf_counter()
            model = load(device, dtype)
            load_s = time.perf_counter() - t0
            result = model.transcribe(str(args.audio), **kwargs)
            break
        except RuntimeError as exc:
            msg = str(exc)
            partial = getattr(exc, "result", None)
            if partial is not None:  # transcription finished but ABC could not be built
                emit(status="error", stage="abc", error=partial.get("abc_error") or msg,
                     warnings=partial.get("warnings", []), device=device)
                return 2
            if "out of memory" in msg.lower() and device == "cuda" and attempt == "primary":
                print(f"[sheetsage] CUDA OOM, retrying on CPU: {msg[:200]}", file=sys.stderr, flush=True)
                del model  # noqa: F821  (may be unbound if load failed; ignore)
                torch.cuda.empty_cache()
                device, dtype = "cpu", "fp32"
                kwargs["dtype"] = "fp32"
                continue
            emit(status="error", stage="transcribe", error=msg, device=device)
            return 1
    else:  # pragma: no cover
        emit(status="error", stage="transcribe", error="unreachable")
        return 1

    abc = result.get("abc")
    emit(status="ok" if abc else "error", device=device, dtype=dtype,
         abc_path=str(args.output / "score.abc") if abc else None,
         abc_error=result.get("abc_error"), warnings=result.get("warnings", []),
         diagnostics=result.get("diagnostics"), peak_gpu_mib=result.get("peak_gpu_mib"),
         num_events=result.get("num_events"), load_seconds=round(load_s, 1),
         total_seconds=round(time.perf_counter() - start, 1))
    return 0 if abc else 2


if __name__ == "__main__":
    raise SystemExit(main())

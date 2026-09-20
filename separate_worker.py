#!/usr/bin/env python3
"""Separate a recording into vocals and accompaniment with Demucs (htdemucs). Runs in .venv-describe.

    YuE/.venv-describe/bin/python separate_worker.py song.mp3 --output out_dir [--model htdemucs] [--device auto|cuda|cpu]

Writes out_dir/vocals.flac and out_dir/no_vocals.flac and prints one JSON line on stdout.
Demucs weights download to the torch hub cache on first use (~80 MB for htdemucs).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def emit(**payload):
    print(json.dumps(payload), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="htdemucs", help="htdemucs (fast, good) or htdemucs_ft (slower, better)")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--shifts", type=int, default=1, help="test-time augmentation passes; 2-5 = better, slower")
    args = parser.parse_args()
    start = time.perf_counter()
    try:
        import torch
        import soundfile as sf
        from demucs.pretrained import get_model
        from demucs.apply import apply_model
        from demucs.audio import convert_audio
        device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        model = get_model(args.model)
        model.to(device).eval()
        # decode with ffmpeg via soundfile fallback
        try:
            data, sr = sf.read(str(args.audio), dtype="float32", always_2d=True)
        except Exception:
            import subprocess, tempfile, os
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-y", "-i", str(args.audio), "-vn", tmp.name], check=True)
                data, sr = sf.read(tmp.name, dtype="float32", always_2d=True)
            os.unlink(tmp.name)
        wav = torch.from_numpy(data.T.copy())  # [C, T]
        wav = convert_audio(wav, sr, model.samplerate, model.audio_channels)
        ref = wav.mean(0)
        wav = (wav - ref.mean()) / (ref.std() + 1e-8)
        with torch.inference_mode():
            sources = apply_model(model, wav[None], device=device, shifts=args.shifts, split=True, overlap=0.25,
                                  progress=False)[0]
        sources = sources * (ref.std() + 1e-8) + ref.mean()
        names = model.sources  # e.g. ['drums', 'bass', 'other', 'vocals']
        vocals = sources[names.index("vocals")]
        rest = sum(sources[i] for i, n in enumerate(names) if n != "vocals")
        args.output.mkdir(parents=True, exist_ok=True)
        out_sr = model.samplerate
        sf.write(str(args.output / "vocals.flac"), vocals.cpu().numpy().T, out_sr, subtype="PCM_24")
        sf.write(str(args.output / "no_vocals.flac"), rest.cpu().numpy().T, out_sr, subtype="PCM_24")
        # a crude "how much singing is there" indicator: fraction of 1 s windows where vocals carry energy
        v = vocals.cpu().numpy()
        frames = v.shape[1] // out_sr
        rms = [float((v[:, i * out_sr:(i + 1) * out_sr] ** 2).mean() ** 0.5) for i in range(frames)]
        full = rest.cpu().numpy()
        rms_all = [float((full[:, i * out_sr:(i + 1) * out_sr] ** 2).mean() ** 0.5) + 1e-9 for i in range(frames)]
        # singing present when the vocal stem is both audible in absolute terms and not just bleed from the band
        active = sum(1 for a, b in zip(rms, rms_all) if a > 0.01 and a > 0.15 * b)
        emit(status="ok", device=device, model=args.model, sample_rate=out_sr,
             vocals=str(args.output / "vocals.flac"), no_vocals=str(args.output / "no_vocals.flac"),
             seconds_total=frames, seconds_with_vocals=active, elapsed=round(time.perf_counter() - start, 1))
        return 0
    except Exception as exc:
        import traceback
        traceback.print_exc()
        emit(status="error", error=str(exc)[:500])
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

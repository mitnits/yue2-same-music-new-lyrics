#!/usr/bin/env python3
"""Gradio web UI for YuE2 (multimodal-art-projection/YuE) song generation.

Tabs:
  * Generate            – lyrics + style -> score -> song
  * Same melody, new lyrics – reuse a previous song's exact score with new words

Usage:
    .venv/bin/python yue2_ui.py [--port 7860] [--host 127.0.0.1] [--budget 24]
                                [--quantization auto|none|fp8] [--offload-ar] [--share]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

import gradio as gr

HERE = Path(__file__).resolve().parent
REPO = HERE / "YuE"
DEFAULT_MODEL = REPO / "models" / "YuE2-3B"
DEFAULT_VAE = REPO / "models" / "YuE2-Vae"
EXAMPLE = json.loads((REPO / "examples" / "song.json").read_text(encoding="utf-8"))
EXAMPLE_MELODY = (REPO / "examples" / "melody.abc").read_text(encoding="utf-8")
EXAMPLE_SCORE = (REPO / "examples" / "score.abc").read_text(encoding="utf-8")

SHEETSAGE_PY = REPO / ".venv-sheetsage2" / "bin" / "python"
SHEETSAGE_WORKER = HERE / "sheetsage_worker.py"
SHEETSAGE_MODEL = REPO / "models" / "SheetSage2"
SEPARATE_WORKER = HERE / "separate_worker.py"  # Demucs, runs in the describe venv
DESCRIBE_PY = REPO / ".venv-describe" / "bin" / "python"
DESCRIBE_WORKER = HERE / "describe_worker.py"
DESCRIBE_MODEL = REPO / "models" / "audio-flamingo-3-hf"
sys.path.insert(0, str(HERE))
from yue2_studio_lib import abc_tools  # noqa: E402  (repo helper: parse_abc / compare / strip_chords)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--model", default=str(DEFAULT_MODEL))
parser.add_argument("--vae", default=str(DEFAULT_VAE))
parser.add_argument("--output-dir", default=str(HERE / "outputs"))
parser.add_argument("--budget", type=float, default=None,
                    help="GPU memory budget in GiB (default: auto from free VRAM)")
parser.add_argument("--quantization", choices=("auto", "none", "fp8"), default="auto",
                    help="auto = fp8 when free VRAM < 22 GiB, else none")
parser.add_argument("--offload-ar", action="store_true", help="forced on in low-VRAM auto mode")
parser.add_argument("--backend", choices=("torch", "torch-eager"), default="torch")
parser.add_argument("--no-verify-hashes", action="store_true")
parser.add_argument("--host", default="127.0.0.1")
parser.add_argument("--port", type=int, default=7860)
parser.add_argument("--share", action="store_true")
ARGS = parser.parse_args()
OUTPUT_DIR = Path(ARGS.output_dir)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------------- runtime
class Runtime:
    """Holds the single loaded pipeline. One generation at a time."""

    def __init__(self):
        self.pipe = None
        self.settings = {}
        self.lock = threading.Lock()
        self.cancel = threading.Event()

    def gpu_status(self):
        try:
            import torch
            if not torch.cuda.is_available():
                return "CUDA not available"
            free, total = torch.cuda.mem_get_info()
            return f"{torch.cuda.get_device_name(0)}: {free / 2**30:.1f} GiB free of {total / 2**30:.1f} GiB"
        except Exception as exc:  # pragma: no cover
            return f"GPU query failed: {exc}"

    def preset(self):
        """Full BF16 preset with >=22 GiB free, otherwise fp8 + AR offload."""
        import torch
        free = torch.cuda.mem_get_info()[0] / 2**30 if torch.cuda.is_available() else 0
        budget, quant, offload = ARGS.budget, ARGS.quantization, ARGS.offload_ar
        low = free < 22
        if budget is None:
            budget = 24 if not low else max(4.0, float(int(free)) + 1.5)
        if quant == "auto":
            quant = "fp8" if low else "none"
        if low:
            offload = True
        return dict(memory_budget_gib=budget, quantization=quant, offload_ar=offload), free

    def status(self):
        if self.pipe is not None:
            st = self.settings
            loaded = (f"loaded ({st['quantization']}, budget {st['memory_budget_gib']:.0f} GiB"
                      f"{', AR offload' if st['offload_ar'] else ''})")
        else:
            loaded = "not loaded (loads on first Generate)"
        note = ""
        try:
            _, free = self.preset()
            if free < 22 and self.pipe is None:
                note = (" | ⚠ under 22 GiB free: low-VRAM preset (fp8 + AR offload) will be used; "
                        "free GPU memory from other apps for the full-quality BF16 preset")
        except Exception:
            pass
        return f"Model: {loaded} | {self.gpu_status()}{note}"

    def load(self, progress=None):
        if self.pipe is not None:
            return self.pipe
        from yue2 import YuE2Pipeline
        settings, free = self.preset()
        self.settings = settings
        if progress:
            progress(0, desc=f"Loading YuE2-3B + VAE ({settings['quantization']}, budget "
                             f"{settings['memory_budget_gib']:.0f} GiB; verifies weights on first load)")
        print(f"[ui] loading pipeline with {settings} (free VRAM {free:.1f} GiB)", flush=True)
        self.pipe = YuE2Pipeline.from_pretrained(
            ARGS.model, vae=ARGS.vae, device="cuda", local_files_only=True,
            backend=ARGS.backend, verify_hashes=not ARGS.no_verify_hashes, **settings,
        )
        return self.pipe

    def unload(self):
        if self.pipe is not None:
            self.pipe.close()
            self.pipe = None
        import gc
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        return self.status()


RT = Runtime()


# ----------------------------------------------------------------------------- helpers
def slugify(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", (text or "").strip()).strip("-.")
    return (text or "song")[:60]


def list_outputs(need_score=False):
    if not OUTPUT_DIR.exists():
        return []
    dirs = [p for p in OUTPUT_DIR.iterdir()
            if (p / "audio.flac").is_file() and (not need_score or (p / "score.abc").is_file())]
    dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return [p.name for p in dirs]


def read_output(name):
    d = OUTPUT_DIR / name
    audio = d / "audio.flac"
    abc = (d / "score.abc").read_text(encoding="utf-8") if (d / "score.abc").is_file() else ""
    req = json.loads((d / "request.json").read_text(encoding="utf-8")) if (d / "request.json").is_file() else {}
    res = json.loads((d / "result.json").read_text(encoding="utf-8")) if (d / "result.json").is_file() else {}
    return (str(audio) if audio.is_file() else None), abc, req, res


from yue2_studio_lib import fit as _fit  # shared with the ComfyUI pack
from yue2_studio_lib import transpose as _transpose
from yue2_studio_lib.fit import (LANGUAGES, vocal_notes_per_section, vocal_phrases_per_section, lyric_sections,
                                 compare_scores, style_with_language, language_note)


def fit_report(abc, original_lyrics, new_lyrics, language="English"):
    return _fit.fit_report(abc, original_lyrics, new_lyrics, language=language)


# ----------------------------------------------------------------------------- core generation
def run_generation(style, lyrics, mode, seed, cfg, abc, song_id, progress):
    """Shared by both tabs. Returns (audio_path, abc_text, info_dict, output_dir)."""
    if not style or not style.strip():
        raise gr.Error("Style prompt is required.")
    if not lyrics or not lyrics.strip():
        raise gr.Error("Lyrics are required.")
    if abc is not None and mode == "off":
        raise gr.Error("A supplied ABC score requires mode 'full' or 'melody'.")
    if not RT.lock.acquire(blocking=False):
        raise gr.Error("A generation is already running. Wait for it or press Stop.")
    try:
        RT.cancel.clear()
        seed = int(seed)
        pipe = RT.load(progress)
        abc_max = pipe.generation_config.abc.max_tokens
        sem_max = pipe.generation_config.semantic.max_tokens
        counts = {"abc": 0, "semantic": 0}
        t0 = time.perf_counter()

        def on_token(phase, _token):
            counts[phase] = counts.get(phase, 0) + 1
            if phase == "abc":
                progress(min(counts["abc"] / abc_max, 0.99) * 0.2, desc=f"Planning score: {counts['abc']} tokens")
            else:
                progress(0.2 + min(counts["semantic"] / sem_max, 0.99) * 0.5,
                         desc=f"Generating song: {counts['semantic']} tokens (~{counts['semantic'] / 25:.0f}s of audio)")

        request = dict(style=style, lyrics=lyrics, cot=mode, seed=seed, id=slugify(song_id))
        if abc is not None:
            request["abc"] = abc
        if cfg is not None:
            request["cfg_scale"] = cfg

        progress(0, desc="Starting")
        plan = pipe.plan(cancelled=RT.cancel.is_set, on_token=on_token, **request)
        semantic = pipe.generate_semantic(plan, cancelled=RT.cancel.is_set, on_token=on_token)
        progress(0.72, desc="Synthesizing acoustic latents (flow matching, 32 steps)")
        latents = pipe.synthesize(semantic, cancelled=RT.cancel.is_set)
        if RT.cancel.is_set():
            raise InterruptedError("Cancelled before VAE")
        progress(0.9, desc="Decoding audio with VAE")
        audio = pipe.decode(latents)
        e2e = time.perf_counter() - t0

        from yue2.pipeline import SongResult
        from yue2.storage import identity
        config = pipe.effective_config(plan.request)
        request_id = identity({"request": plan.request.to_dict(), "config": config, "weights": pipe.weights})
        timing = {"abc": plan.timing, "semantic": semantic.timing, "e2e_seconds": e2e, "load": dict(pipe.load_timing)}
        song = SongResult(audio, 48000, semantic, latents, config, pipe.weights, timing, request_id)

        stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        out = OUTPUT_DIR / f"{stamp}_{request['id']}_seed{seed}"
        result = song.save_artifacts(out)
        progress(1.0, desc="Done")
        info = {"output_dir": str(out), "seed": seed, "mode": mode, "cfg_scale": plan.request.guidance,
                "score_supplied": abc is not None, "audio_seconds": round(result["audio_seconds"], 1),
                "generation_seconds": round(e2e, 1), "truncated": result["truncated"]}
        return str(out / "audio.flac"), song.abc or "", info, out
    except InterruptedError:
        raise gr.Error("Generation cancelled.")
    except gr.Error:
        raise
    except Exception as exc:
        traceback.print_exc()
        msg = str(exc)
        if "out of memory" in msg.lower():
            msg += " | GPU is out of memory. Free VRAM used by other processes, press 'Unload model', and try again."
        raise gr.Error(msg)
    finally:
        RT.lock.release()


def stop():
    RT.cancel.set()
    return "Stop requested; the current stage will abort shortly."


# ----------------------------------------------------------------------------- tab 1: generate
def generate(style, lyrics, mode, seed, random_seed, cfg_scale, abc_text, abc_file, song_id, progress=gr.Progress()):
    abc = None
    if abc_file:
        abc = Path(abc_file).read_text(encoding="utf-8")
    elif abc_text and abc_text.strip():
        abc = abc_text
    if random_seed:
        seed = random.randrange(0, 2**31)
    cfg = None if cfg_scale is None or float(cfg_scale) <= 0 else float(cfg_scale)
    audio, abc_out, info, out = run_generation(style, lyrics, mode, seed, cfg, abc, song_id or "song", progress)
    return (audio, abc_out, json.dumps(info, indent=2),
            gr.update(choices=list_outputs(), value=out.name),
            gr.update(choices=list_outputs(need_score=True)), RT.status())


def load_output(name):
    if not name:
        return None, "", "{}"
    audio, abc, req, res = read_output(name)
    return audio, abc, json.dumps({"request.json": req, "result.json": res}, indent=2)


# ----------------------------------------------------------------------------- tab 2: same melody, new lyrics
def load_source(name, language="English"):
    if not name:
        raise gr.Error("Pick a source song first.")
    audio, abc, req, res = read_output(name)
    if not abc.strip():
        raise gr.Error("That output has no score.abc (it was generated with planning mode 'off'). "
                       "Only songs generated in 'full' or 'melody' mode carry a reusable score.")
    style = style_with_language(req.get("style", ""), language) if req.get("style") else language
    lyrics = req.get("lyrics", "")
    seed = req.get("seed", EXAMPLE["seed"])
    base = re.sub(r"^\d{8}-\d{6}_", "", name)
    base = re.sub(r"_seed\d+$", "", base)
    return (audio, abc, style, lyrics, lyrics, seed, f"{base}_newlyrics",
            fit_report(abc, lyrics, lyrics, language))


def load_source_file(path, current_lyrics, language="English"):
    if not path:
        return gr.update(), gr.update()
    abc = Path(path).read_text(encoding="utf-8")
    return abc, fit_report(abc, "", current_lyrics, language)


def transpose_preview(source_abc, key_shift, vocal_octave):
    """Live readout under the sliders: key and vocal range before/after."""
    if not source_abc or not source_abc.strip():
        return ""
    try:
        new, info = _transpose.transpose(source_abc, int(key_shift), int(vocal_octave))
    except Exception as exc:
        return f"⚠ cannot transpose: {exc}"
    if not info.get("changed"):
        return f"No transposition. Vocal range **{_transpose.vocal_range(source_abc)}**; {_transpose.voice_type_hint(source_abc)}."
    return (f"Key **{info['key']}** (shift {info['key_shift']:+d}), vocal octave {info['vocal_octave']:+d}. "
            f"Vocal range {info['vocal_range_before']} → **{info['vocal_range_after']}**; {_transpose.voice_type_hint(new)}. "
            "Match the style prompt's voice type to the new range.")


def regenerate(source_abc, new_lyrics, style, seed, keep_chords, song_id, source_name, language="English",
               key_shift=0, vocal_octave=0, progress=gr.Progress()):
    style = style_with_language(style, language)
    transposed_note = ""
    if source_abc and source_abc.strip() and (int(key_shift) or int(vocal_octave)):
        try:
            source_abc, tinfo = _transpose.transpose(source_abc, int(key_shift), int(vocal_octave))
            transposed_note = f"transposed: key {tinfo['key']}, vocal octave {tinfo['vocal_octave']:+d}, range {tinfo['vocal_range_after']}"
        except Exception as exc:
            raise gr.Error(f"Transposition failed: {exc}")
    if not source_abc or not source_abc.strip():
        raise gr.Error("No source score. Load a previous song or upload its score.abc.")
    try:
        abc_tools.parse_abc(source_abc)
    except Exception as exc:
        raise gr.Error(f"The source score is not in YuE2's native ABC dialect: {exc}")
    if keep_chords:
        abc, mode = source_abc, "full"
    else:
        abc, mode = abc_tools.strip_chords(source_abc), "melody"
    audio, abc_out, info, out = run_generation(style, new_lyrics, mode, seed, None, abc, song_id or "newlyrics", progress)
    # Record provenance next to the artifacts.
    prov = {"source_output": source_name, "kept_chords": keep_chords, "source_score_sha256": abc_tools.hashlib.sha256(
        source_abc.encode()).hexdigest()}
    (out / "same_melody_source.json").write_text(json.dumps(prov, indent=2), encoding="utf-8")
    check = compare_scores(source_abc, abc_out)
    info["score_check"] = check
    if transposed_note:
        info["transposition"] = transposed_note
    return (audio, check, json.dumps(info, indent=2),
            gr.update(choices=list_outputs(), value=out.name),
            gr.update(choices=list_outputs(need_score=True)), RT.status())


# ----------------------------------------------------------------------------- SheetSage2 transcription
def sheetsage_available():
    return SHEETSAGE_PY.is_file() and SHEETSAGE_WORKER.is_file() and (SHEETSAGE_MODEL / "config.json").is_file()


def transcribe_recording(audio_path, melody_only, name, language="English", isolate_vocals=False,
                         progress=gr.Progress()):
    """Run SheetSage2 (separate venv) on a recording and store it as a source entry in outputs/."""
    if not audio_path:
        raise gr.Error("Upload a recording first.")
    if not sheetsage_available():
        raise gr.Error("SheetSage2 is not installed (expected YuE/.venv-sheetsage2 and YuE/models/SheetSage2).")
    if not RT.lock.acquire(blocking=False):
        raise gr.Error("A generation is already running. Wait for it to finish first.")
    try:
        src = Path(audio_path)
        slug = slugify(name or src.stem)
        stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        out = OUTPUT_DIR / f"{stamp}_transcribed_{slug}"
        out.mkdir(parents=True)
        progress(0.05, desc="Converting recording to 48 kHz FLAC")
        conv = subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-y", "-i", str(src), "-vn", "-ar", "48000",
                               "-ac", "2", str(out / "audio.flac")], capture_output=True, text=True)
        if conv.returncode != 0:
            shutil.rmtree(out, ignore_errors=True)
            raise gr.Error(f"ffmpeg could not read the recording: {conv.stderr.strip()[:300]}")
        sep_note = ""
        source_for_sheetsage = out / "audio.flac"
        if isolate_vocals:
            if not (DESCRIBE_PY.is_file() and SEPARATE_WORKER.is_file()):
                raise gr.Error("Vocal isolation needs Demucs in YuE/.venv-describe and separate_worker.py.")
            progress(0.1, desc="Demucs: separating vocals from the band")
            sp = subprocess.run([str(DESCRIBE_PY), str(SEPARATE_WORKER), str(out / "audio.flac"), "--output",
                                 str(out / "stems")], capture_output=True, text=True, cwd=str(HERE))
            sl = [ln for ln in sp.stdout.splitlines() if ln.startswith("{")]
            sep = json.loads(sl[-1]) if sl else {"status": "error", "error": sp.stderr[-400:]}
            if sep.get("status") != "ok":
                raise gr.Error(f"Demucs failed: {sep.get('error')}")
            import soundfile as sf
            v, sr = sf.read(sep["vocals"], dtype="float32", always_2d=True)
            r, _ = sf.read(sep["no_vocals"], dtype="float32", always_2d=True)
            n = min(len(v), len(r)); mix = v[:n] + 0.25 * r[:n]
            sf.write(str(out / "vocal_forward_mix.flac"), mix / max(float(abs(mix).max()) or 1.0, 1.0), sr, subtype="PCM_24")
            source_for_sheetsage = out / "vocal_forward_mix.flac"
            sep_note = (f"Demucs: **{sep['seconds_with_vocals']} of {sep['seconds_total']} s** contain singing"
                        + (" — ⚠ almost none: this recording looks instrumental." if sep["seconds_with_vocals"] < 5 else ".")
                        + "  \n")
        progress(0.3, desc="SheetSage2: loading model and transcribing (melody, chords, beats, key, structure)")
        cmd = [str(SHEETSAGE_PY), str(SHEETSAGE_WORKER), str(source_for_sheetsage), "--output", str(out / "sheetsage")]
        if melody_only:
            cmd.append("--melody-only")
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(HERE))
        lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("{")]
        (out / "sheetsage.log").write_text(proc.stdout + "\n--- stderr ---\n" + proc.stderr, encoding="utf-8")
        report = json.loads(lines[-1]) if lines else {"status": "error", "error": proc.stderr.strip()[-600:]}
        if report.get("status") != "ok":
            raise gr.Error(f"Transcription failed: {report.get('error') or report.get('abc_error')} "
                           f"(see {out / 'sheetsage.log'})")
        abc = (out / "sheetsage" / "score.abc").read_text(encoding="utf-8")
        try:
            sc0 = abc_tools.parse_abc(abc)
            if not sc0.voices["Vocal"].notes and sc0.voices["Ins"].notes:
                from yue2_studio_lib import voices as _voices
                abc, vnotes = _voices.transform(abc, "ins_as_vocal")
                sep_note += "⚠ SheetSage2 filed the melody under the instrument track; moved it to Vocal (" + vnotes[0] + ")  \n"
            native = "✅ score is valid YuE2-native ABC"
        except Exception as exc:
            native = f"⚠ score did not pass the native-dialect check: {exc}"
        cleaned_note = ""
        (out / "score.abc").write_text(abc, encoding="utf-8")
        bpm = re.search(r"^Q:1/4=(\d+)", abc, re.M)
        style_hint = f"{language}, <genre>, <vocal character>, <instruments>, {bpm.group(1)} BPM" if bpm else language
        seed = random.randrange(0, 2**31)
        mode = "melody" if melody_only else "full"
        json.dump({"style": "", "lyrics": "", "cot": mode, "seed": seed, "id": slug, "abc": None,
                   "source": "sheetsage2", "original_audio": src.name},
                  open(out / "request.json", "w"), indent=2)
        json.dump({"status": "transcribed", "tool": "SheetSage2", "melody_only": melody_only, "report": report,
                   "seconds": round(time.perf_counter() - t0, 1)}, open(out / "result.json", "w"), indent=2)
        progress(1.0, desc="Done")
        sung = vocal_notes_per_section(abc)
        sections = ", ".join(f"{s}: {n} notes" for s, n in sung if n > 0) or "no sung notes found"
        warns = report.get("warnings") or []
        msg = (sep_note + f"**Transcribed with SheetSage2 on {report.get('device')} in {report.get('total_seconds')} s.** {native}.  \n"
               f"Vocal sections → {sections}.  \n" + cleaned_note
               + (f"Warnings: {'; '.join(map(str, warns))}  \n" if warns else "")
               + f"Saved as `{out.name}` (score.abc, MIDI and annotations under `sheetsage/`). "
               "Review the score below, paste the original lyrics if you know them, write a style prompt, then add new lyrics.")
        return (str(out / "audio.flac"), abc, style_hint, "", "", seed, f"{slug}_newlyrics",
                fit_report(abc, "", "", language), msg,
                gr.update(choices=list_outputs(need_score=True), value=out.name),
                gr.update(choices=list_outputs(), value=out.name))
    except gr.Error:
        raise
    except Exception as exc:
        traceback.print_exc()
        raise gr.Error(str(exc))
    finally:
        RT.lock.release()


# ----------------------------------------------------------------------------- optional: describe the track
def describe_available():
    return DESCRIBE_PY.is_file() and DESCRIBE_WORKER.is_file() and (DESCRIBE_MODEL / "config.json").is_file()


def describe_track(audio_path, source_abc, source_name, language, progress=gr.Progress()):
    """Run Audio Flamingo 3 + CLAP on the source recording and draft a YuE2 style prompt. Optional feature."""
    if not audio_path:
        raise gr.Error("Load a source song first (its audio is what gets described).")
    if not describe_available():
        raise gr.Error("The describer is not installed (expected YuE/.venv-describe and YuE/models/audio-flamingo-3-hf).")
    if not RT.lock.acquire(blocking=False):
        raise gr.Error("A generation is already running. Wait for it to finish first.")
    try:
        unloaded = ""
        if RT.pipe is not None:
            RT.unload()
            unloaded = "YuE2 was unloaded to make room for the describer; it reloads on the next Generate.  \n"
        lab = OUTPUT_DIR / source_name / "sheetsage" / "structure.lab" if source_name else None
        facts = _fit.score_facts(source_abc or "", str(lab) if lab else None)
        progress(0.1, desc="Audio Flamingo 3 + CLAP listening to the track (1-3 min)")
        out_json = None
        if source_name and (OUTPUT_DIR / source_name).is_dir():
            out_json = OUTPUT_DIR / source_name / "description.json"
        cmd = [str(DESCRIBE_PY), str(DESCRIBE_WORKER), str(audio_path), "--facts", json.dumps(facts)]
        if out_json:
            cmd += ["--output", str(out_json)]
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(HERE))
        lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("{")]
        report = json.loads(lines[-1]) if lines else {"status": "error", "error": proc.stderr.strip()[-800:]}
        if report.get("status") != "ok":
            raise gr.Error(f"Describer failed: {report.get('error')}")
        progress(1.0, desc="Done")
        md = [f"**Described in {report.get('seconds')} s.** {unloaded}"]
        if facts:
            md.append("**Measured (SheetSage2):** " + "; ".join(f"{k}: {v}" for k, v in facts.items()))
        for key, q in (("instruments", "Instruments"), ("drums", "Drums / rhythm"), ("vocal", "Lead vocal"),
                       ("genre", "Genre, era, production"), ("mood", "Mood and arc")):
            if key in report.get("answers", {}):
                md.append(f"**{q}:** {report['answers'][key]}")
        clap = report.get("clap") or {}
        if clap:
            rows = ["| CLAP group | top matches (relative score) |", "|---|---|"]
            for group, items in clap.items():
                rows.append(f"| {group} | " + ", ".join(f"{name} {score:.2f}" for name, score in items[:4]) + " |")
            md.append("Cross-check with CLAP zero-shot tags (relative within each group, averaged over "
                      f"{report.get('clap_windows')} windows):\n\n" + "\n".join(rows))
        cons = report.get("consensus") or []
        if cons:
            rows = ["| Instrument | open answer | yes/no check | CLAP | votes |", "|---|---|---|---|---|"]
            for t in sorted(cons, key=lambda t: (-t["votes"], -t["clap"])):
                if t["votes"] == 0:
                    continue
                mark = lambda b: "✔" if b else "·"
                rows.append(f"| {t['instrument']} | {mark(t['open_answer'])} | {mark(t['yes_no_check'])} | "
                            f"{t['clap'] if t['clap'] else '·'} | **{t['votes']}** |")
            md.append("**Instrument consensus** (kept in the prompt when 2 of 3 signals agree):\n\n" + "\n".join(rows))
        if report.get("style_prompt"):
            md.append(f"Audio Flamingo's own free-form line (less reliable, for comparison): _{report['style_prompt']}_")
        md.append("_Listening models mishear instruments (this one says yes too easily). Compare with your own ears "
                  "and edit the suggested prompt before using it. The first tag is the language of the NEW lyrics "
                  "(from the selector), not of the recording._")
        style = style_with_language(report.get("style_prompt_consensus") or report.get("style_prompt", ""), language)
        return "\n\n".join(md), style
    except gr.Error:
        raise
    except Exception as exc:
        traceback.print_exc()
        raise gr.Error(str(exc))
    finally:
        RT.lock.release()


# ----------------------------------------------------------------------------- UI
with gr.Blocks(title="YuE2 Music Generation") as demo:
    gr.Markdown("# 🎵 YuE2 — lyrics + style → editable score → full song")
    status_box = gr.Markdown(RT.status())

    with gr.Tabs():
        # ---------------------------------------------------------------- Generate
        with gr.Tab("Generate"):
            gr.Markdown(
                "`full` plans melody and chords, `melody` plans melody only, `off` generates directly from text. "
                "Paste or upload an ABC score to render your own composition.")
            with gr.Row():
                with gr.Column(scale=1):
                    style = gr.Textbox(label="Style / tags", lines=3, value=EXAMPLE["style"],
                                       placeholder="Language, genre, instruments, vocal character, tempo…")
                    lyrics = gr.Textbox(label="Lyrics (use [Verse], [Chorus] … tags)", lines=14, value=EXAMPLE["lyrics"])
                    with gr.Row():
                        mode = gr.Radio(["full", "melody", "off"], value="full", label="Planning mode (cot)")
                        song_id = gr.Textbox(label="Song name", value=EXAMPLE["id"])
                    with gr.Row():
                        seed = gr.Number(label="Seed", value=EXAMPLE["seed"], precision=0)
                        random_seed = gr.Checkbox(label="Random seed", value=False)
                        cfg_scale = gr.Slider(0, 5, value=0, step=0.05, label="CFG scale (0 = model default)")
                    with gr.Accordion("Custom ABC score (cover / edit)", open=False):
                        gr.Markdown("Requires mode `full` (melody + chords) or `melody` (no chord symbols).")
                        abc_text = gr.Textbox(label="ABC score", lines=10, placeholder="X:1\nT:\nM:4/4\n...")
                        abc_file = gr.File(label="…or upload .abc file", file_types=[".abc", ".txt"], type="filepath")
                        with gr.Row():
                            gr.Button("Load example melody.abc", size="sm").click(
                                lambda: (EXAMPLE_MELODY, "melody"), outputs=[abc_text, mode])
                            gr.Button("Load example score.abc", size="sm").click(
                                lambda: (EXAMPLE_SCORE, "full"), outputs=[abc_text, mode])
                            gr.Button("Clear score", size="sm").click(lambda: ("", None), outputs=[abc_text, abc_file])
                    with gr.Row():
                        go = gr.Button("🎶 Generate", variant="primary")
                        stop_btn = gr.Button("⏹ Stop")
                        unload_btn = gr.Button("Unload model (free VRAM)")
                with gr.Column(scale=1):
                    audio_out = gr.Audio(label="Generated song (48 kHz stereo FLAC)", type="filepath")
                    abc_out = gr.Textbox(label="Generated / used score (ABC)", lines=14)
                    info_out = gr.Code(label="Run info", language="json")
                    with gr.Accordion("Previous outputs", open=True):
                        history = gr.Dropdown(choices=list_outputs(), label="Output folder", interactive=True,
                                              allow_custom_value=True)
                        with gr.Row():
                            gr.Button("Refresh list", size="sm").click(
                                lambda: gr.update(choices=list_outputs()), outputs=history)
                            gr.Button("Use this score as input", size="sm").click(
                                lambda s: s, inputs=abc_out, outputs=abc_text)
                        history.change(load_output, inputs=history, outputs=[audio_out, abc_out, info_out])

        # ---------------------------------------------------------------- Same melody, new lyrics
        with gr.Tab("Same melody, new lyrics"):
            gr.Markdown(
                "Take a song you already generated, keep its **exact score** (every vocal and instrumental note, "
                "rhythm, tempo, and chords), and sing **new words** over it. YuE2 has no lyric-only inpainting: the "
                "score is the melody's white-box handle, so the composition is identical while the vocal "
                "performance is re-rendered. Keep the same style and seed for the closest sound.\n\n"
                "Fit the words to the notes: each `[Section]` in the new lyrics should have about as many "
                "syllables as that section has sung notes (see the fit table).")
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Row():
                        src_pick = gr.Dropdown(choices=list_outputs(need_score=True),
                                               label="Source song (previous output with a score)", interactive=True,
                                               allow_custom_value=True)
                        gr.Button("Refresh", size="sm").click(
                            lambda: gr.update(choices=list_outputs(need_score=True)), outputs=src_pick)
                    load_btn = gr.Button("Load source song", variant="secondary")
                    src_file = gr.File(label="…or upload a score.abc from elsewhere", file_types=[".abc", ".txt"],
                                       type="filepath")
                    with gr.Accordion("…or transcribe a recording with SheetSage2 (song from anywhere)",
                                      open=sheetsage_available()):
                        gr.Markdown(
                            "SheetSage2 listens to a recording and writes its vocal melody, instrumental melody, "
                            "chords, tempo and sections as a YuE2 score. Transcriptions can contain errors: "
                            "review the score before rendering." if sheetsage_available() else
                            "⚠ SheetSage2 is not installed.")
                        tr_audio = gr.Audio(label="Recording (mp3, wav, flac, m4a…)", type="filepath")
                        with gr.Row():
                            tr_name = gr.Textbox(label="Name", placeholder="my_song")
                            tr_melody_only = gr.Checkbox(value=False,
                                                         label="Melody only (drop transcribed chords; render in mode melody)")
                            tr_isolate = gr.Checkbox(value=False,
                                                     label="Isolate vocals first (Demucs): use when the singer gets filed as an instrument")
                        tr_btn = gr.Button("🎼 Transcribe to score", variant="secondary",
                                           interactive=sheetsage_available())
                        tr_status = gr.Markdown()
                    src_audio = gr.Audio(label="Source song (for reference)", type="filepath")
                    src_lyrics = gr.Textbox(label="Original lyrics (reference; paste them for a transcribed song)",
                                            lines=10, interactive=True)
                    src_abc = gr.Textbox(label="Source score (ABC) — reused verbatim", lines=10)
                with gr.Column(scale=1):
                    with gr.Row():
                        s_language = gr.Dropdown(LANGUAGES, value="English", label="Language of the NEW lyrics",
                                                 info="Becomes the first style tag and picks the syllable counter. "
                                                      "YuE2-3B documents Chinese and English; others are best effort.")
                        lang_note = gr.Markdown()
                    new_lyrics = gr.Textbox(label="New lyrics (same [Section] order as the original)", lines=14)
                    fit_md = gr.Markdown("_Load a source song to see the syllable fit table._")
                    with gr.Row():
                        s_style = gr.Textbox(label="Style (kept from source; edit to change vocal/arrangement)", lines=2)
                        s_seed = gr.Number(label="Seed (kept from source)", precision=0)
                    with gr.Accordion("Optional: describe the source track → draft a style prompt (Audio Flamingo 3)",
                                      open=False):
                        gr.Markdown(
                            "Listens to the loaded source recording and drafts YuE2 style tags: instruments and how "
                            "they are played, drums, vocal, genre/era, mood, plus the measured tempo and key. "
                            "Uses about 17 GB of VRAM, so YuE2 is unloaded while it runs." if describe_available() else
                            "⚠ Describer not installed (Audio Flamingo 3 + CLAP).")
                        desc_btn = gr.Button("🎧 Describe the track", variant="secondary",
                                             interactive=describe_available())
                        desc_md = gr.Markdown()
                        desc_style = gr.Textbox(label="Suggested style prompt (edit freely)", lines=2)
                        gr.Button("Use as style", size="sm").click(lambda t: t, inputs=desc_style, outputs=s_style)
                    with gr.Row():
                        keep_chords = gr.Checkbox(value=True, label="Keep chord symbols (exact harmony; mode full)")
                        s_id = gr.Textbox(label="Song name")
                    with gr.Accordion("Range levers: vocal octave / key shift (0 = off)", open=True):
                        with gr.Row():
                            s_vocal_oct = gr.Slider(-2, 2, value=0, step=1, label="Vocal octave (singer only; same notes, lower/higher)")
                            s_key_shift = gr.Slider(-11, 11, value=0, step=1, label="Key shift in semitones (whole song, chords included)")
                        transpose_md = gr.Markdown()
                    with gr.Row():
                        s_go = gr.Button("🎶 Generate with new lyrics", variant="primary")
                        s_stop = gr.Button("⏹ Stop")
                    s_audio = gr.Audio(label="New song", type="filepath")
                    s_check = gr.Markdown()
                    s_info = gr.Code(label="Run info", language="json")

            load_btn.click(load_source, inputs=[src_pick, s_language],
                           outputs=[src_audio, src_abc, s_style, src_lyrics, new_lyrics, s_seed, s_id, fit_md])
            src_file.change(load_source_file, inputs=[src_file, new_lyrics, s_language], outputs=[src_abc, fit_md])
            tr_btn.click(transcribe_recording, inputs=[tr_audio, tr_melody_only, tr_name, s_language, tr_isolate],
                         outputs=[src_audio, src_abc, s_style, src_lyrics, new_lyrics, s_seed, s_id, fit_md,
                                  tr_status, src_pick, history])
            src_lyrics.change(fit_report, inputs=[src_abc, src_lyrics, new_lyrics, s_language], outputs=fit_md)
            desc_btn.click(describe_track, inputs=[src_audio, src_abc, src_pick, s_language], outputs=[desc_md, desc_style])
            new_lyrics.change(fit_report, inputs=[src_abc, src_lyrics, new_lyrics, s_language], outputs=fit_md)
            s_language.change(lambda a, o, n, lg, st: (fit_report(a, o, n, lg), language_note(lg),
                                                       style_with_language(st, lg) if st else st),
                              inputs=[src_abc, src_lyrics, new_lyrics, s_language, s_style],
                              outputs=[fit_md, lang_note, s_style])
            for comp in (s_vocal_oct, s_key_shift, src_abc):
                comp.change(transpose_preview, inputs=[src_abc, s_key_shift, s_vocal_oct], outputs=transpose_md)
            s_go.click(regenerate,
                       inputs=[src_abc, new_lyrics, s_style, s_seed, keep_chords, s_id, src_pick, s_language,
                               s_key_shift, s_vocal_oct],
                       outputs=[s_audio, s_check, s_info, history, src_pick, status_box])
            s_stop.click(stop, outputs=status_box)

    go.click(generate,
             inputs=[style, lyrics, mode, seed, random_seed, cfg_scale, abc_text, abc_file, song_id],
             outputs=[audio_out, abc_out, info_out, history, src_pick, status_box])
    stop_btn.click(stop, outputs=status_box)
    unload_btn.click(RT.unload, outputs=status_box)
    demo.load(RT.status, outputs=status_box)

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(
        server_name=ARGS.host, server_port=ARGS.port, share=ARGS.share, allowed_paths=[str(OUTPUT_DIR)],
        show_error=True)

"""yue2-same-music-new-lyrics nodes: the pieces around ComfyUI's native YuE2 / SheetSage2 nodes.

Native ComfyUI already provides: CheckpointLoaderSimple (yue2_3b_*.safetensors), YuE2GenerateABC,
YuE2GenerateMusic (accepts an ABC score), EmptyYuE2LatentAudio, KSampler, VAEDecodeAudio,
AudioEncoderLoader + SheetSage2AudioToABC (transcription).

This pack adds what the Gradio UI does for the "same melody, new lyrics" and "song from anywhere" flows:
  * Transcribe (full SheetSage2 release, via its own venv): ABC + section timeline + key/tempo facts
  * Describe Track (Audio Flamingo 3 + CLAP, own venv): a draft YuE2 style prompt from the recording
  * Lyrics Fit: syllables per lyric section vs sung notes per phrase in the score
  * Score Facts / Strip Chords / Compare Scores / Transpose: ABC utilities
  * Load Audio (path) / Load Text (path): convenience loaders

The two heavy tools run as subprocesses in environments that setup.sh creates under runtime/ (override with
YUE2_RUNTIME), so ComfyUI's own torch/transformers versions never conflict with theirs.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import os
import re
import subprocess
import tempfile
from pathlib import Path

import torch
from typing_extensions import override

import comfy.model_management
import folder_paths
from comfy_api.latest import ComfyExtension, io

from .lib import abc_tools, fit, transpose as transpose_lib

HERE = Path(__file__).resolve().parent
RUNTIME = Path(os.environ.get("YUE2_RUNTIME", HERE / "runtime")).expanduser()  # created by setup.sh
SHEETSAGE_PY = RUNTIME / ".venv-sheetsage2" / "bin" / "python"
SHEETSAGE_WORKER = HERE / "workers" / "sheetsage_worker.py"
DESCRIBE_PY = RUNTIME / ".venv-describe" / "bin" / "python"
DESCRIBE_WORKER = HERE / "workers" / "describe_worker.py"
DESCRIBE_MODEL = RUNTIME / "models" / "audio-flamingo-3-hf"
CATEGORY = "audio/yue2-same-music-new-lyrics"


# ----------------------------------------------------------------------------- helpers
def _work_dir(prefix: str) -> Path:
    base = Path(folder_paths.get_output_directory()) / "yue2_studio"
    base.mkdir(parents=True, exist_ok=True)
    d = base / f"{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}_{prefix}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _write_audio(audio: dict, path: Path) -> Path:
    """ComfyUI AUDIO = {"waveform": [B, C, T] tensor, "sample_rate": int}. Writes the first batch item as FLAC."""
    import soundfile as sf
    wav = audio["waveform"]
    if wav.ndim == 3:
        wav = wav[0]
    data = wav.detach().float().cpu().transpose(0, 1).numpy()  # [T, C]
    sf.write(str(path), data, int(audio["sample_rate"]), subtype="PCM_24")
    return path


def _run_worker(cmd: list[str], log_path: Path) -> dict:
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(HERE),
                          env={**os.environ, "YUE2_RUNTIME": str(RUNTIME)})
    log_path.write_text(proc.stdout + "\n--- stderr ---\n" + proc.stderr, encoding="utf-8")
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("{")]
    if not lines:
        raise RuntimeError(f"Worker produced no result (exit {proc.returncode}); see {log_path}\n"
                           f"{proc.stderr.strip()[-800:]}")
    report = json.loads(lines[-1])
    if report.get("status") != "ok":
        raise RuntimeError(f"Worker failed: {report.get('error') or report.get('abc_error')} (see {log_path})")
    return report


def _require(path: Path, what: str):
    if not path.exists():
        raise RuntimeError(f"{what} not found at {path}. Run setup.sh in the node folder (or set YUE2_RUNTIME).")


# ----------------------------------------------------------------------------- nodes
class Yue2SmlTranscribe(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Yue2SmlTranscribe",
            display_name="Transcribe Recording (SheetSage2 full)",
            category=CATEGORY,
            description=("Runs the full SheetSage2 release in its own environment: ABC score with section comments, "
                         "chords, key, tempo, plus a section timeline and MIDI/annotation files saved under "
                         "output/yue2_studio. Use ComfyUI's native 'SheetSage2 Audio to ABC' for a lighter, in-process "
                         "transcription."),
            inputs=[
                io.Audio.Input("audio"),
                io.Boolean.Input("melody_only", default=False,
                                 tooltip="Drop chord symbols (then render with mode 'melody')."),
                io.String.Input("name", default="song"),
            ],
            outputs=[
                io.String.Output(display_name="abc"),
                io.String.Output(display_name="facts_json"),
                io.String.Output(display_name="summary"),
                io.String.Output(display_name="folder"),
            ],
        )

    @classmethod
    def execute(cls, audio, melody_only, name):
        _require(SHEETSAGE_PY, "SheetSage2 environment (runtime/.venv-sheetsage2)")
        _require(SHEETSAGE_WORKER, "sheetsage_worker.py")
        out = _work_dir(f"transcribed_{re.sub(r'[^A-Za-z0-9_.-]+', '-', name) or 'song'}")
        wav = _write_audio(audio, out / "audio.flac")
        cmd = [str(SHEETSAGE_PY), str(SHEETSAGE_WORKER), str(wav), "--output", str(out / "sheetsage")]
        if melody_only:
            cmd.append("--melody-only")
        report = _run_worker(cmd, out / "sheetsage.log")
        abc = (out / "sheetsage" / "score.abc").read_text(encoding="utf-8")
        (out / "score.abc").write_text(abc, encoding="utf-8")
        facts = fit.score_facts(abc, str(out / "sheetsage" / "structure.lab"))
        phrases = fit.vocal_phrases_per_section(abc)
        try:
            abc_tools.parse_abc(abc)
            valid = "valid YuE2-native ABC"
        except Exception as exc:  # pragma: no cover
            valid = f"WARNING native-dialect check failed: {exc}"
        summary = (f"SheetSage2 ({report.get('device')}, {report.get('total_seconds')} s): {valid}. "
                   + "; ".join(f"{k}: {v}" for k, v in facts.items() if k != "structure") + ". Sung sections: "
                   + ", ".join(f"{s} {sum(p)} notes ({'/'.join(map(str, p))})" for s, p in phrases)
                   + (f". Warnings: {report.get('warnings')}" if report.get("warnings") else ""))
        return io.NodeOutput(abc, json.dumps(facts, ensure_ascii=False), summary, str(out))


class Yue2SmlDescribe(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Yue2SmlDescribe",
            display_name="Describe Track → Style Prompt (Audio Flamingo 3 + CLAP)",
            category=CATEGORY,
            description=("Listens to the recording and drafts a YuE2 style line: genre, vocal, instruments (kept when "
                         "2 of 3 signals agree), drums, mood, plus measured key and tempo from the ABC. Needs ~17 GB "
                         "VRAM: other models are unloaded first. Optional; edit the result before use."),
            inputs=[
                io.Audio.Input("audio"),
                io.String.Input("abc", default="", multiline=True, optional=True,
                                tooltip="Score of the same recording, for measured tempo/key/meter."),
                io.Combo.Input("language", options=fit.LANGUAGES, default="English",
                               tooltip="Language of the lyrics you will sing (becomes the first style tag)."),
                io.Boolean.Input("use_audio_flamingo", default=True),
                io.Boolean.Input("use_clap", default=True),
            ],
            outputs=[
                io.String.Output(display_name="style_prompt"),
                io.String.Output(display_name="report"),
                io.String.Output(display_name="af3_free_line"),
            ],
        )

    @classmethod
    def execute(cls, audio, language, use_audio_flamingo, use_clap, abc=""):
        _require(DESCRIBE_PY, "Describer environment (runtime/.venv-describe)")
        _require(DESCRIBE_WORKER, "describe_worker.py")
        comfy.model_management.unload_all_models()
        comfy.model_management.soft_empty_cache()
        out = _work_dir("described")
        wav = _write_audio(audio, out / "audio.flac")
        facts = fit.score_facts(abc or "")
        cmd = [str(DESCRIBE_PY), str(DESCRIBE_WORKER), str(wav), "--facts", json.dumps(facts),
               "--output", str(out / "description.json")]
        if not use_audio_flamingo:
            cmd.append("--no-af3")
        if not use_clap:
            cmd.append("--no-clap")
        report = _run_worker(cmd, out / "describe.log")
        style = fit.style_with_language(report.get("style_prompt_consensus") or report.get("style_prompt") or "", language)
        md = [f"Described in {report.get('seconds')} s."]
        if facts:
            md.append("Measured: " + "; ".join(f"{k}: {v}" for k, v in facts.items()))
        for key, label in (("instruments", "Instruments"), ("drums", "Drums"), ("vocal", "Vocal"),
                           ("genre", "Genre/production"), ("mood", "Mood")):
            if key in report.get("answers", {}):
                md.append(f"{label}: {report['answers'][key]}")
        for group, items in (report.get("clap") or {}).items():
            md.append(f"CLAP {group}: " + ", ".join(f"{n} {s:.2f}" for n, s in items[:4]))
        cons = [t for t in report.get("consensus") or [] if t["votes"]]
        if cons:
            md.append("Instrument votes (open/yes-no/CLAP): " + ", ".join(
                f"{t['instrument']}={t['votes']}" for t in sorted(cons, key=lambda t: -t["votes"])))
        md.append("Listening models mishear instruments; edit the prompt against your own ears.")
        return io.NodeOutput(style, "\n\n".join(md), report.get("style_prompt", ""))


class Yue2SmlLyricsFit(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Yue2SmlLyricsFit",
            display_name="Lyrics Fit Check",
            category=CATEGORY,
            description=("Compares syllables per [Section] of the new lyrics with sung notes per 4-bar phrase in the "
                         "score (and with the original lyrics when given). Note counts are a floor: transcriptions snap "
                         "to an eighth-note grid, so fast syllable pairs merge into one note."),
            inputs=[
                io.String.Input("abc", multiline=True),
                io.String.Input("new_lyrics", multiline=True),
                io.Combo.Input("language", options=fit.LANGUAGES, default="English",
                               tooltip="Language of the NEW lyrics; picks the syllable counter (Cyrillic and CJK are exact "
                                       "or near-exact, Hebrew/Arabic are rough estimates). YuE2-3B documents Chinese and English only."),
                io.Combo.Input("method", options=fit.METHODS, default="auto",
                               tooltip="Override the counter; 'off' compares only section/line structure."),
                io.String.Input("original_lyrics", default="", multiline=True, optional=True),
                io.Boolean.Input("fail_on_mismatch", default=False,
                                 tooltip="ON: stop the workflow when the lyrics' section count differs from the score "
                                         "(saves a wasted render). OFF (default): only log the mismatch to the console "
                                         "and the report, and let the render continue."),
            ],
            outputs=[
                io.String.Output(display_name="report"),
                io.Boolean.Output(display_name="sections_match"),
                io.Int.Output(display_name="sung_notes"),
                io.Int.Output(display_name="new_syllables"),
            ],
        )

    @classmethod
    def execute(cls, abc, new_lyrics, language, method, fail_on_mismatch, original_lyrics=""):
        m = method if method != "auto" else fit.method_for(language)
        report = fit.fit_report(abc, original_lyrics or "", new_lyrics or "", language=language, method=m)
        sung = fit.vocal_phrases_per_section(abc)
        new = fit.lyric_sections(new_lyrics or "", m)
        match = len(sung) == len(new)
        if not (new_lyrics or "").strip():
            report = ("_No lyrics yet. The score has " + str(len(sung)) + " sung sections: "
                      + ", ".join(f"{s} ({sum(p)} notes)" for s, p in sung)
                      + ". Paste lyrics with one [Section] per sung section, in this order._\n\n" + report)
        elif not match:
            msg = (f"Lyrics Fit Check: the score has {len(sung)} sung sections but the lyrics have {len(new)} "
                   f"sections with words ({', '.join(f'{s}:{sum(p)}' for s, p in sung)} vs "
                   f"{', '.join(f'[{t}]:{n}' for t, _, n, _ in new)}).")
            if fail_on_mismatch:
                raise RuntimeError(msg + " Match the section count/order, or turn off fail_on_mismatch to render anyway.")
            logging.warning("[yue2-sml] " + msg + " Rendering anyway (fail_on_mismatch is off).")
        else:
            logging.info(f"[yue2-sml] Lyrics Fit Check: {len(sung)} sections match; "
                         f"{sum(sum(p) for _, p in sung)} sung notes vs {sum(x[2] for x in new)} syllables.")
        return io.NodeOutput(report, match, sum(sum(p) for _, p in sung), sum(x[2] for x in new))


class Yue2SmlScoreEditor(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Yue2SmlScoreEditor",
            display_name="Score Editor (edit ABC in place)",
            category=CATEGORY,
            description=("The pause between transcribing and rendering. The box fills with the incoming score and "
                         "refreshes whenever a NEW score arrives, as long as you have not edited it. Once you edit "
                         "the box, your text is what flows downstream (queue again; upstream nodes are cached). "
                         "Type RESET or clear the box to go back to the incoming score. Saves the score that went "
                         "downstream to output/yue2_studio/<name>.abc."),
            inputs=[
                io.String.Input("abc", default="", multiline=True, optional=True, force_input=True,
                                tooltip="Incoming score (from Transcribe or Load Text)."),
                io.String.Input("edited_abc", default="", multiline=True,
                                tooltip="Your editable copy. Auto-filled; edit it to override the incoming score."),
                io.String.Input("name", default="edited_score"),
                io.String.Input("autofilled_hash", default="", tooltip="Managed by the editor; leave alone."),
            ],
            outputs=[io.String.Output(display_name="abc"), io.String.Output(display_name="saved_path"),
                     io.String.Output(display_name="source")],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, edited_abc, name, abc="", autofilled_hash=""):
        import hashlib
        digest = lambda t: hashlib.sha256((t or "").strip().encode("utf-8")).hexdigest()[:16]
        incoming = abc or ""
        box = (edited_abc or "").strip()
        user_edited = bool(box) and box != "RESET" and digest(box) != (autofilled_hash or "")
        if user_edited:
            text, source, fill = edited_abc, "edited text (box differs from the auto-filled score)", edited_abc
        elif incoming.strip():
            text, source, fill = incoming, "incoming score (box auto-filled)", incoming
        elif box and box != "RESET":
            text, source, fill = edited_abc, "box text (nothing connected upstream)", edited_abc
        else:
            raise RuntimeError("Score Editor: no score. Connect a score to 'abc' or paste one into 'edited_abc'.")
        try:
            abc_tools.parse_abc(text)
        except Exception as exc:
            raise RuntimeError(f"Score Editor: the score is not valid YuE2-native ABC: {exc}")
        base = Path(folder_paths.get_output_directory()) / "yue2_studio"
        base.mkdir(parents=True, exist_ok=True)
        path = base / f"{re.sub(r'[^A-Za-z0-9_.-]+', '-', name) or 'edited_score'}.abc"
        path.write_text(text, encoding="utf-8")
        logging.info(f"[yue2-sml] Score Editor used the {source}; saved {path}")
        return io.NodeOutput(text, str(path), source,
                             ui={"text": (text,), "fill": (fill,), "hash": (digest(fill),)})


class Yue2SmlTranspose(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Yue2SmlTranspose",
            display_name="Transpose (key shift / vocal octave)",
            category=CATEGORY,
            description=("Two safe levers for a singer's range. key_shift moves the whole song (both voices, chords, key "
                         "signature) by semitones, like playing it in another key. vocal_octave moves only the sung "
                         "line by whole octaves (same notes, lower or higher register). Both keep the harmony intact. "
                         "The report shows the vocal range before and after; update the style prompt's voice "
                         "description (e.g. baritone vs tenor) to match."),
            inputs=[
                io.String.Input("abc", multiline=True),
                io.Int.Input("key_shift", default=0, min=-11, max=11, display_mode=io.NumberDisplay.slider,
                             tooltip="Semitones for the whole song: -2 = two semitones lower (a whole tone)."),
                io.Int.Input("vocal_octave", default=0, min=-2, max=2, display_mode=io.NumberDisplay.slider,
                             tooltip="Octaves for the sung line only: -1 = the singer one octave down."),
            ],
            outputs=[io.String.Output(display_name="abc"), io.String.Output(display_name="report")],
        )

    @classmethod
    def execute(cls, abc, key_shift, vocal_octave):
        new, info = transpose_lib.transpose(abc, key_shift, vocal_octave)
        if not info.get("changed"):
            rng = transpose_lib.vocal_range(abc)
            return io.NodeOutput(abc, f"No transposition. Vocal range {rng}; {transpose_lib.voice_type_hint(abc)}.")
        rep = (f"Key {info['key']} (shift {info['key_shift']:+d} semitones), vocal octave {info['vocal_octave']:+d}. "
               f"Vocal range {info['vocal_range_before']} → {info['vocal_range_after']}; "
               f"{transpose_lib.voice_type_hint(new)}.")
        logging.info("[yue2-sml] " + rep)
        return io.NodeOutput(new, rep)


class Yue2SmlScoreFacts(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Yue2SmlScoreFacts",
            display_name="Score Facts (tempo, key, sections)",
            category=CATEGORY,
            inputs=[io.String.Input("abc", multiline=True)],
            outputs=[
                io.Int.Output(display_name="bpm"),
                io.String.Output(display_name="key"),
                io.String.Output(display_name="meter"),
                io.String.Output(display_name="sections"),
                io.Float.Output(display_name="nominal_seconds"),
            ],
        )

    @classmethod
    def execute(cls, abc):
        facts = fit.score_facts(abc)
        bpm = int(facts.get("tempo", "0 BPM").split()[0]) if facts.get("tempo") else 0
        seconds = 0.0
        try:
            score = abc_tools.parse_abc(abc)
            seconds = float(score.voices["Vocal"].time * 60 / score.bpm)
        except Exception:
            pass
        sections = ", ".join(f"{s}: {sum(p)} notes ({'/'.join(map(str, p))})"
                             for s, p in fit.vocal_phrases_per_section(abc))
        return io.NodeOutput(bpm, facts.get("key", ""), facts.get("meter", ""), sections, seconds)


class Yue2SmlStripChords(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Yue2SmlStripChords",
            display_name="Strip Chords from Score",
            category=CATEGORY,
            description="Removes chord symbols so YuE2 can re-harmonize (render with mode 'melody'). Optionally keeps one voice.",
            inputs=[
                io.String.Input("abc", multiline=True),
                io.Combo.Input("keep_voice", options=["both", "Vocal", "Ins"], default="both"),
            ],
            outputs=[io.String.Output(display_name="abc")],
        )

    @classmethod
    def execute(cls, abc, keep_voice):
        return io.NodeOutput(abc_tools.strip_chords(abc, keep_voice=keep_voice))


class Yue2SmlCompareScores(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Yue2SmlCompareScores",
            display_name="Compare Scores (melody preserved?)",
            category=CATEGORY,
            description="Exact symbolic check that notes, timing, meter and tempo are unchanged between two ABC scores.",
            inputs=[
                io.String.Input("before", multiline=True),
                io.String.Input("after", multiline=True),
                io.Boolean.Input("allow_tempo_change", default=False),
            ],
            outputs=[io.Boolean.Output(display_name="match"), io.String.Output(display_name="report")],
        )

    @classmethod
    def execute(cls, before, after, allow_tempo_change):
        try:
            rep = abc_tools.compare(abc_tools.parse_abc(before), abc_tools.parse_abc(after),
                                    allow_tempo_change=allow_tempo_change)
        except Exception as exc:
            return io.NodeOutput(False, f"Could not compare: {exc}")
        text = ("MATCH: melody, rhythm, meter and tempo identical." if rep.get("match")
                else "DIFFERENCES: " + "; ".join(rep.get("differences", [])))
        return io.NodeOutput(bool(rep.get("match")), text)


class Yue2SmlStyleLanguage(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Yue2SmlStyleLanguage",
            display_name="Set Song Language (style prompt)",
            category=CATEGORY,
            description=("Makes the chosen language the first tag of the style prompt (YuE2's only language control) "
                         "and returns a caveat when the language is outside the documented Chinese/English."),
            inputs=[
                io.String.Input("style", multiline=True),
                io.Combo.Input("language", options=fit.LANGUAGES, default="English"),
            ],
            outputs=[io.String.Output(display_name="style"), io.String.Output(display_name="note")],
        )

    @classmethod
    def execute(cls, style, language):
        return io.NodeOutput(fit.style_with_language(style, language), fit.language_note(language) or "ok")


AUDIO_EXT = {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".opus", ".aac", ".wma", ".aiff", ".aif"}


def audio_folders():
    """Folders scanned for the file dropdown: ComfyUI/input, ~/Downloads, ~/Music, plus YUE2_AUDIO_DIRS (colon-separated)."""
    dirs = [Path(folder_paths.get_input_directory()), Path.home() / "Downloads", Path.home() / "Music"]
    dirs += [Path(d).expanduser() for d in os.environ.get("YUE2_AUDIO_DIRS", "").split(":") if d]
    return [d for d in dirs if d.is_dir()]


def audio_choices():
    """'<folder name>/<file>' entries, newest first inside each folder. Press R in ComfyUI to refresh the list."""
    out = []
    for d in audio_folders():
        files = [f for f in d.iterdir() if f.is_file() and f.suffix.lower() in AUDIO_EXT]
        files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        out += [f"{d.name}/{f.name}" for f in files]
    return out or ["(no audio files found in input, Downloads, Music)"]


def resolve_choice(choice: str) -> Path | None:
    for d in audio_folders():
        if choice.startswith(d.name + "/"):
            p = d / choice[len(d.name) + 1:]
            if p.is_file():
                return p
    return None


class Yue2SmlLoadAudioPath(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Yue2SmlLoadAudioPath",
            display_name="Load Audio (file chooser: input / Downloads / Music)",
            category=CATEGORY,
            description=("Pick a recording from ComfyUI/input, ~/Downloads or ~/Music (newest first; press R to refresh "
                         "the list after adding files), or type any absolute path in 'path' to override. For "
                         "browser upload use the core 'Load Audio' node instead. Extra folders: YUE2_AUDIO_DIRS."),
            inputs=[
                io.Combo.Input("file", options=audio_choices()),
                io.String.Input("path", default="", optional=True,
                                tooltip="Optional absolute path; when set it wins over the dropdown."),
            ],
            outputs=[io.Audio.Output(display_name="audio")],
        )

    @classmethod
    def fingerprint_inputs(cls, file, path=""):
        p = Path(path).expanduser() if path else resolve_choice(file)
        return (str(p), p.stat().st_mtime if p and p.is_file() else 0)

    @classmethod
    def execute(cls, file, path=""):
        p = Path(path).expanduser() if path else resolve_choice(file)
        if not p or not p.is_file():
            raise RuntimeError(f"No such audio file: {path or file}. Press R in ComfyUI to refresh the file list.")
        import soundfile as sf
        try:
            data, sr = sf.read(str(p), dtype="float32", always_2d=True)  # [T, C]
        except Exception:  # formats libsndfile cannot read (e.g. m4a): decode with ffmpeg first
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-y", "-i", str(p), "-vn", tmp.name], check=True)
                data, sr = sf.read(tmp.name, dtype="float32", always_2d=True)
            os.unlink(tmp.name)
        wav = torch.from_numpy(data.T.copy())  # [C, T]
        return io.NodeOutput({"waveform": wav.unsqueeze(0), "sample_rate": int(sr)})


class Yue2SmlLoadText(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Yue2SmlLoadText",
            display_name="Load Text File (lyrics / ABC)",
            category=CATEGORY,
            inputs=[io.String.Input("path", default="yue2_studio/edited_score.abc",
                                    tooltip="Absolute path, or relative to ComfyUI/output.")],
            outputs=[io.String.Output(display_name="text")],
        )

    @classmethod
    def execute(cls, path):
        p = Path(path).expanduser()
        if not p.is_absolute():
            p = Path(folder_paths.get_output_directory()) / p  # e.g. yue2_studio/edited_score.abc
        if not p.is_file():
            raise RuntimeError(f"No such file: {p}")
        return io.NodeOutput(p.read_text(encoding="utf-8"))


class Yue2SmlExtension(ComfyExtension):
    @override
    async def get_node_list(self):
        return [Yue2SmlTranscribe, Yue2SmlScoreEditor, Yue2SmlTranspose,
                Yue2SmlDescribe,
                Yue2SmlLyricsFit, Yue2SmlScoreFacts, Yue2SmlStripChords, Yue2SmlCompareScores,
                Yue2SmlStyleLanguage, Yue2SmlLoadAudioPath, Yue2SmlLoadText]


async def comfy_entrypoint():
    return Yue2SmlExtension()

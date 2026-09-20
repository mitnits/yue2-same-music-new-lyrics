#!/usr/bin/env python3
"""Describe a recording for prompt writing: Audio Flamingo 3 Q&A + CLAP zero-shot tags. Runs in .venv-describe.

    YuE/.venv-describe/bin/python describe_worker.py song.mp3 --output desc.json [--facts '{"bpm":91,...}']

Prints one JSON line on stdout: {"status": "ok", "answers": {...}, "style_prompt": "...", "clap": {...}, ...}
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
AF3_DIR = HERE / "YuE" / "models" / "audio-flamingo-3-hf"
CLAP_DIR = HERE / "YuE" / "models" / "clap-music"

# Open questions carry NO examples: audio-language models tend to echo example lists back as "heard" instruments.
QUESTIONS = [
    ("instruments", "Which instruments can you hear in this recording? For each one, say what it plays (rhythm, "
                    "chords, lead, bass line) and how (e.g. its playing technique and tone). Name only what you hear."),
    ("drums", "Is there a drummer, a drum machine, light percussion, or no drums? Describe the rhythm pattern."),
    ("vocal", "Describe the lead singer: gender, range, and delivery. Are there backing vocals? Do not transcribe words."),
    ("genre", "Which genre, sub-genre and decade does this sound like, and what does the production sound like "
              "(studio or live, lo-fi or hi-fi, reverb, stereo width)?"),
    ("mood", "Describe the mood and energy, and how they change from intro to verses, chorus, solo and ending."),
]

# Closed checks: yes/no per candidate is far more reliable than an open list.
CHECKS = [
    "acoustic guitar", "clean electric guitar", "distorted electric guitar", "electric bass", "piano", "electric piano",
    "organ", "synthesizer or synth pad", "string section (violins, cellos)", "saxophone", "trumpet or brass",
    "flute", "harmonica", "accordion", "drum kit", "drum machine", "backing vocals or harmonies",
]


def emit(**payload):
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def load_audio(path, sr):
    import librosa
    y, _ = librosa.load(str(path), sr=sr, mono=True)
    return y


def _prep(processor, conversation, model):
    import torch
    inputs = processor.apply_chat_template(conversation, tokenize=True, add_generation_prompt=True,
                                           return_dict=True).to(model.device)
    for k, v in list(inputs.items()):  # audio features come back float32; the model runs in bf16
        if torch.is_tensor(v) and v.is_floating_point():
            inputs[k] = v.to(model.dtype)
    return inputs


def run_af3(audio_path, facts, max_seconds):
    import torch
    from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor
    processor = AutoProcessor.from_pretrained(str(AF3_DIR))
    model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
        str(AF3_DIR), dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    sr = processor.feature_extractor.sampling_rate
    y = load_audio(audio_path, sr)
    if max_seconds and len(y) > max_seconds * sr:
        y = y[: int(max_seconds * sr)]
    conversation = []
    answers = {}
    for i, (key, question) in enumerate(QUESTIONS):
        content = [{"type": "text", "text": question}]
        if i == 0:
            content.append({"type": "audio", "audio": y})
        conversation.append({"role": "user", "content": content})
        inputs = _prep(processor, conversation, model)
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=300, do_sample=False)
        text = processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
        answers[key] = text
        conversation.append({"role": "assistant", "content": [{"type": "text", "text": text}]})
        print(f"[af3] {key}: {text[:120]}", file=sys.stderr, flush=True)

    # Verification pass: one yes/no per candidate instrument, judged fresh against the audio each time.
    verified = {}
    base = conversation[:1]  # the first user turn holds the audio
    for cand in CHECKS:
        q = (f"Is {cand} clearly audible in this recording? Answer with exactly one word, yes or no, "
             "then one short sentence of evidence.")
        conv = base + [{"role": "user", "content": [{"type": "text", "text": q}]}]
        inputs = _prep(processor, conv, model)
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=60, do_sample=False)
        text = processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
        verified[cand] = {"yes": text.lower().lstrip(" *\"'").startswith("yes"), "evidence": text}
        print(f"[af3] check {cand}: {text[:90]}", file=sys.stderr, flush=True)
    present = [c for c, v in verified.items() if v["yes"]]

    facts_txt = ", ".join(f"{k}: {v}" for k, v in (facts or {}).items()) or "none"
    final = ("Write ONE line of comma-separated style tags for a song generation model that must recreate this "
             "recording's arrangement. Use only these confirmed elements: instruments = {present}; vocal = your "
             "vocal answer; genre/era and mood = your answers; measured facts = {facts} (trust the measured key and "
             "tempo over your own guess). Order: language of the vocal, genre and era, lead vocal character, each "
             "instrument with how it is played, drums, mood, production feel, tempo as 'NN BPM'. Under 60 words. "
             "Output only the line.").format(present=", ".join(present) or "unknown", facts=facts_txt)
    conversation.append({"role": "user", "content": [{"type": "text", "text": final}]})
    inputs = _prep(processor, conversation, model)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=120, do_sample=False)
    style = processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
    style = style.strip('"').splitlines()[0].strip() if style else ""
    answers["verified_instruments"] = verified
    del model
    torch.cuda.empty_cache()
    return answers, style


CLAP_VOCAB = {
    "instruments": ["acoustic guitar", "electric guitar", "distorted electric guitar", "clean electric guitar",
                    "bass guitar", "piano", "electric piano", "organ", "synthesizer", "strings", "violin", "cello",
                    "saxophone", "trumpet", "flute", "harmonica", "accordion", "mandolin", "banjo", "harp"],
    "drums": ["drum kit", "drum machine", "hand percussion", "no drums"],
    "vocal": ["male vocal", "female vocal", "spoken word", "choir", "instrumental without vocals"],
    "genre": ["rock", "folk rock", "art rock", "post-punk", "pop", "indie rock", "blues", "jazz", "reggae",
              "punk rock", "hard rock", "singer-songwriter", "new wave", "ballad", "psychedelic rock"],
    "mood": ["melancholic", "energetic", "calm", "dark", "uplifting", "nostalgic", "aggressive", "dreamy"],
}


def _embed(out):
    """Transformers 4.x returns a tensor; 5.x returns an output object with the (normalized) pooled embedding."""
    import torch
    emb = out if torch.is_tensor(out) else out.pooler_output
    return torch.nn.functional.normalize(emb, dim=-1)


def run_clap(audio_path, window=10.0, hop=30.0):
    import numpy as np
    import torch
    from transformers import ClapModel, ClapProcessor
    processor = ClapProcessor.from_pretrained(str(CLAP_DIR))
    model = ClapModel.from_pretrained(str(CLAP_DIR)).eval().to("cuda")
    sr = processor.feature_extractor.sampling_rate
    y = load_audio(audio_path, sr)
    wins = [y[int(s * sr): int((s + window) * sr)] for s in np.arange(0, max(len(y) / sr - window, 1e-6), hop)]
    wins = [w for w in wins if len(w) >= sr]
    with torch.inference_mode():
        a = processor(audio=wins, sampling_rate=sr, return_tensors="pt", padding=True).to("cuda")
        audio_emb = _embed(model.get_audio_features(**a))
        result = {}
        for group, labels in CLAP_VOCAB.items():
            t = processor(text=[f"This is a recording of {x} music" if group in ("genre", "mood") else
                                f"This music features {x}" for x in labels], return_tensors="pt", padding=True).to("cuda")
            text_emb = _embed(model.get_text_features(**t))
            sims = (audio_emb @ text_emb.T).mean(0)  # average over windows
            probs = torch.softmax(sims * 100 / 3, dim=-1)  # sharpened relative scores within the group
            order = torch.argsort(probs, descending=True)
            result[group] = [(labels[i], round(float(probs[i]), 3)) for i in order[:6]]
    del model
    torch.cuda.empty_cache()
    return result, len(wins)


# Map closed-check names -> (keywords in the open answer, CLAP labels)
CONSENSUS_MAP = {
    "acoustic guitar": (["acoustic guitar"], ["acoustic guitar"]),
    "clean electric guitar": (["clean electric", "electric guitar", "lead guitar", "rhythm guitar", "guitar"], ["clean electric guitar", "electric guitar"]),
    "distorted electric guitar": (["distort", "overdriv", "power chord"], ["distorted electric guitar"]),
    "electric bass": (["bass"], ["bass guitar"]),
    "piano": (["piano"], ["piano"]),
    "electric piano": (["electric piano", "rhodes"], ["electric piano"]),
    "organ": (["organ"], ["organ"]),
    "synthesizer or synth pad": (["synth"], ["synthesizer"]),
    "string section (violins, cellos)": (["string", "violin", "cello"], ["strings", "violin", "cello"]),
    "saxophone": (["sax"], ["saxophone"]),
    "trumpet or brass": (["trumpet", "brass", "horn"], ["trumpet"]),
    "flute": (["flute"], ["flute"]),
    "harmonica": (["harmonica"], ["harmonica"]),
    "accordion": (["accordion"], ["accordion"]),
    "drum kit": (["drummer", "drum kit", "kick", "snare", "hi-hat"], ["drum kit"]),
    "drum machine": (["drum machine", "programmed", "electronic drum"], ["drum machine"]),
}
DISPLAY = {"clean electric guitar": "clean electric guitar", "synthesizer or synth pad": "synth pad",
           "string section (violins, cellos)": "strings", "trumpet or brass": "brass"}


def build_consensus(payload, facts):
    """Vote per instrument: open answer mention, yes/no check, CLAP rank within its group (top-3 with score>=0.05)."""
    answers = payload["answers"]
    open_text = (answers.get("instruments", "") + " " + answers.get("drums", "")).lower()
    checks = answers.get("verified_instruments", {})
    clap = payload.get("clap") or {}
    clap_hits = {}
    for group in ("instruments", "drums"):
        for rank, (label, score) in enumerate(clap.get(group, [])):
            if rank < 3 and score >= 0.05:
                clap_hits[label] = score
    table = []
    for name, (keys, clap_labels) in CONSENSUS_MAP.items():
        in_open = any(k in open_text for k in keys)
        in_check = bool(checks.get(name, {}).get("yes"))
        clap_score = max((clap_hits.get(l, 0.0) for l in clap_labels), default=0.0)
        votes = int(in_open) + int(in_check) + int(clap_score > 0)
        table.append({"instrument": name, "open_answer": in_open, "yes_no_check": in_check,
                      "clap": round(clap_score, 2), "votes": votes})
    present = [t for t in table if t["votes"] >= 2]
    # drums: kit vs machine vs none decided by the strongest evidence
    drums = [t for t in present if t["instrument"] in ("drum kit", "drum machine")]
    drums_txt = ""
    if drums:
        d = max(drums, key=lambda t: (t["votes"], t["clap"]))
        drums_txt = "steady drum kit" if d["instrument"] == "drum kit" else "drum machine beat"
    elif (clap.get("drums") or [("", 0)])[0][0] == "no drums":
        drums_txt = "no drums"
    inst = [DISPLAY.get(t["instrument"], t["instrument"]) for t in present
            if t["instrument"] not in ("drum kit", "drum machine")]
    genre = (clap.get("genre") or [("rock", 0)])[0][0]
    mood = ", ".join(l for l, sc in (clap.get("mood") or [])[:2] if sc >= 0.1) or "melancholic"
    vocal = answers.get("vocal", "")
    vl = vocal.lower()
    voice = "male" if "male" in vl and "female" not in vl else ("female" if "female" in vl else "lead")
    rng = next((r for r in ("baritone", "tenor", "bass", "alto", "soprano") if r in vl), "")
    delivery = next((w for w in ("half-spoken", "spoken", "raspy", "breathy", "powerful", "intimate", "confident", "emotive")
                     if w in vl), "expressive")
    tempo = facts.get("tempo", "")
    key = facts.get("key", "")
    parts = ["English", f"{genre}", f"{voice} {rng} vocal, {delivery} delivery".replace("  ", " ")]
    parts += inst
    if drums_txt:
        parts.append(drums_txt)
    parts.append(mood)
    if key:
        parts.append(f"key of {key.replace('m', ' minor') if key.endswith('m') else key + ' major'}")
    if tempo:
        parts.append(tempo)
    return table, ", ".join(p for p in parts if p)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--facts", default="{}", help="JSON with measured facts (bpm, key, meter, structure)")
    parser.add_argument("--max-seconds", type=float, default=600)
    parser.add_argument("--no-clap", action="store_true")
    parser.add_argument("--no-af3", action="store_true")
    args = parser.parse_args()
    facts = json.loads(args.facts)
    start = time.perf_counter()
    payload = {"status": "ok", "audio": str(args.audio), "facts": facts}
    try:
        if not args.no_clap:
            payload["clap"], payload["clap_windows"] = run_clap(args.audio)
        if not args.no_af3:
            payload["answers"], payload["style_prompt"] = run_af3(args.audio, facts, args.max_seconds)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        payload.update(status="error", error=str(exc)[:500])
    if payload["status"] == "ok" and "answers" in payload:
        payload["consensus"], payload["style_prompt_consensus"] = build_consensus(payload, facts)
    payload["seconds"] = round(time.perf_counter() - start, 1)
    if args.output:
        args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    emit(**payload)
    return 0 if payload["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())

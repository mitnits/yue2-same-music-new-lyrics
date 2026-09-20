"""Transposition for YuE2-native ABC scores.

Two musically safe levers:
  * key_shift (semitones): move EVERYTHING (both voices, chord symbols, key signature). Harmony is preserved;
    this is "playing the song in another key" for a singer's range.
  * vocal_octave (octaves): move only the Vocal voice by whole octaves. Same notes in a lower/higher register,
    always consonant with the band.

Pitches are resolved exactly like abc_tools.parse (key signature, bar-local accidentals propagated by letter
across octaves, ties), shifted, then re-spelled in the new key with explicit accidentals wherever the reader
would otherwise infer a different alteration. The result is validated with abc_tools and cross-checked note
by note against the source.
"""
from __future__ import annotations

import re

from . import abc_tools
from .abc_tools import KEYS, NATURAL, TOKEN, key_accidentals

PITCH = re.compile(r"(?P<root>[A-G])(?P<acc>bb|##|b|#)?(?P<quality>.*?)(?:/(?P<broot>[A-G])(?P<bacc>bb|##|b|#)?)?$")
ACC_TEXT = {0: "", 1: "^", -1: "_", 2: "^^", -2: "__"}
ACC_NAME = {0: "", 1: "#", -1: "b", 2: "##", -2: "bb"}
NAMES_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
NAMES_FLAT = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]


def _key_count(key: str) -> int:
    if key not in KEYS:
        raise ValueError(f"Unsupported key {key!r}")
    return KEYS[key]


def transpose_key(key: str, semitones: int) -> str:
    """New key name after `semitones`, same mode, preferring the spelling with fewer accidentals (sharps on a tie)."""
    minor = key.endswith("m")
    count = _key_count(key)
    # each semitone up = +7 on the circle of fifths
    raw = count + 7 * (semitones % 12)
    candidates = [c for c in range(-7, 8) if (c - raw) % 12 == 0]  # same pitch class, any enharmonic spelling
    best = sorted(candidates, key=lambda c: (abs(c), -c))[0]
    for name, c in KEYS.items():
        if c == best and name.endswith("m") == minor:
            return name
    raise ValueError(f"No key for count {best}")


def _spell(pitch: int, key_alt: dict[str, int], prefer_sharps: bool) -> tuple[str, int]:
    """Choose (letter, alteration) for a MIDI pitch: diatonic spelling if the key has one, else ±1 by preference."""
    pc = pitch % 12
    for letter, base in NATURAL.items():
        if (base + key_alt[letter]) % 12 == pc:
            return letter, key_alt[letter]
    order = (1, -1, 2, -2) if prefer_sharps else (-1, 1, -2, 2)
    for alt in order:
        for letter, base in NATURAL.items():
            if (base + alt) % 12 == pc:
                return letter, alt
    raise ValueError("unspellable pitch")  # pragma: no cover


def _emit_note(pitch: int, alteration: int, letter: str, tie: str, duration: str, local: dict, key_alt: dict) -> str:
    inferred = local.get(letter, key_alt[letter])
    acc = ""
    if alteration != inferred:
        acc = ACC_TEXT[alteration] or "="
        local[letter] = alteration
    written = pitch - alteration
    diff = written - (60 + NATURAL[letter])
    if diff % 12:
        raise ValueError("spelling mismatch")  # pragma: no cover
    if diff >= 12:
        name = letter.lower() + "'" * (diff // 12 - 1)
    else:
        name = letter + "," * (-diff // 12)
    return f"{acc}{name}{duration}{tie}"


def _transpose_chord(chord: str, semitones: int, prefer_sharps: bool) -> str:
    m = PITCH.match(chord)
    if not m:
        return chord
    names = NAMES_SHARP if prefer_sharps else NAMES_FLAT

    def move(root, acc):
        pc = (NATURAL[root] + {None: 0, "#": 1, "b": -1, "##": 2, "bb": -2}[acc] + semitones) % 12
        return names[pc]
    out = move(m.group("root"), m.group("acc")) + (m.group("quality") or "")
    if m.group("broot"):
        out += "/" + move(m.group("broot"), m.group("bacc"))
    return out


def _transpose_line(line: str, shift: int, chord_shift: int, key_in: str, key_out: str, prefer_sharps: bool):
    """Transpose one music line (ends with '|'). Returns (new_line, key_in_after, key_out_after)."""
    alt_in, alt_out = key_accidentals(key_in), key_accidentals(key_out)
    bars_out = []
    pending = None  # (old_pitch, old_written, spelled letter, alteration) for tie continuations
    for bar in line[:-1].split("|"):
        bar = bar.strip()
        if re.fullmatch(r"Z([2-4])?", bar):
            bars_out.append(bar)
            pending = None
            continue
        local_in, local_out = {}, {}
        cursor, out = 0, []
        while cursor < len(bar):
            if bar[cursor].isspace():
                cursor += 1
                continue
            m = TOKEN.match(bar, cursor)
            if m is None:
                raise ValueError(f"unsupported token at {bar[cursor:cursor + 20]!r}")
            cursor = m.end()
            if m.group("chord") is not None:
                out.append('"' + _transpose_chord(m.group("chord"), chord_shift, prefer_sharps) + '"')
                continue
            if m.group("key") is not None:
                key_in = m.group("key")
                key_out = transpose_key(key_in, shift)
                alt_in, alt_out = key_accidentals(key_in), key_accidentals(key_out)
                local_in, local_out = {}, {}
                out.append(f"[K:{key_out}]")
                continue
            note, acc, octave, duration, tie = m.group("note", "acc", "oct", "duration", "tie")
            if note == "z":
                out.append(m.group(0))
                pending = None
                continue
            letter = note.upper()
            written = 60 + NATURAL[letter] + (12 if note.islower() else 0) + 12 * (octave.count("'") - octave.count(","))
            alteration = local_in.get(letter, alt_in[letter])
            if acc:
                alteration = {"=": 0, "_": -1, "__": -2, "^": 1, "^^": 2}[acc]
                local_in[letter] = alteration
            pitch = written + alteration
            continuation = pending is not None and not acc and written == pending[1]
            if continuation:
                pitch = pending[0]
            new_pitch = pitch + shift
            if not 0 <= new_pitch <= 127:
                raise ValueError(f"pitch {new_pitch} outside MIDI range after transposition")
            if continuation:
                new_letter, new_alt = pending[2], pending[3]
            else:
                new_letter, new_alt = _spell(new_pitch, alt_out, prefer_sharps)
            out.append(_emit_note(new_pitch, new_alt, new_letter, tie, duration, local_out, alt_out))
            pending = (pitch, written, new_letter, new_alt) if tie else None
        bars_out.append("".join(out))
    return "|".join(bars_out) + "|", key_in, key_out


def transpose(abc: str, key_shift: int = 0, vocal_octave: int = 0) -> tuple[str, dict]:
    """Return (new_abc, info). key_shift moves everything; vocal_octave additionally moves the Vocal voice by octaves."""
    key_shift = int(key_shift)
    vocal_octave = int(vocal_octave)
    if key_shift == 0 and vocal_octave == 0:
        return abc, {"changed": False}
    lines = abc.splitlines()
    if len(lines) < 12 or not lines[7].startswith("K:"):
        raise ValueError("Not a YuE2-native two-voice ABC score")
    key0 = lines[7][2:].strip()
    key0_out = transpose_key(key0, key_shift)
    prefer_sharps = _key_count(key0_out) >= 0
    out = lines[:7] + [f"K:{key0_out}"]
    key_in = {"Vocal": key0, "Ins": key0}
    key_out = {"Vocal": key0_out, "Ins": key0_out}
    voice = None
    for ln in lines[8:]:
        if ln.startswith("V: "):
            voice = ln[3:].strip()
            out.append(ln)
        elif ln.startswith("K:"):
            key_in[voice] = ln[2:].strip()
            key_out[voice] = transpose_key(key_in[voice], key_shift)
            out.append(f"K:{key_out[voice]}")
        elif ln.startswith(("% ", "M:")) or not ln.strip():
            out.append(ln)
        else:
            shift = key_shift + (12 * vocal_octave if voice == "Vocal" else 0)
            new_line, key_in[voice], key_out[voice] = _transpose_line(ln, shift, key_shift, key_in[voice],
                                                                       key_out[voice], prefer_sharps)
            out.append(new_line)
    new = "\n".join(out) + ("\n" if abc.endswith("\n") else "")
    before, after = abc_tools.parse_abc(abc), abc_tools.parse_abc(new)
    for name, extra in (("Vocal", 12 * vocal_octave), ("Ins", 0)):
        a, b = before.voices[name].notes, after.voices[name].notes
        if len(a) != len(b) or any(y[1] - x[1] != key_shift + extra or x[0] != y[0] or x[2] != y[2]
                                   for x, y in zip(a, b)):
            raise ValueError(f"internal check failed for {name}: transposition changed something other than pitch")
    info = {"changed": True, "key": f"{key0} → {key0_out}", "key_shift": key_shift, "vocal_octave": vocal_octave,
            "vocal_range_before": vocal_range(abc), "vocal_range_after": vocal_range(new)}
    return new, info


def midi_name(p: int) -> str:
    return f"{NAMES_SHARP[p % 12]}{p // 12 - 1}"


def vocal_range(abc: str) -> str:
    notes = abc_tools.parse_abc(abc).voices["Vocal"].notes
    if not notes:
        return "no sung notes"
    lo, hi = min(n[1] for n in notes), max(n[1] for n in notes)
    return f"{midi_name(lo)}–{midi_name(hi)} (MIDI {lo}–{hi})"


VOICE_TYPES = [  # rough comfortable ranges, for the report only
    ("bass", 40, 60), ("baritone", 45, 65), ("tenor", 48, 69), ("alto", 53, 74), ("mezzo-soprano", 57, 79),
    ("soprano", 60, 84),
]


def voice_type_hint(abc: str) -> str:
    notes = abc_tools.parse_abc(abc).voices["Vocal"].notes
    if not notes:
        return ""
    lo, hi = min(n[1] for n in notes), max(n[1] for n in notes)
    fits = [name for name, a, b in VOICE_TYPES if a <= lo and hi <= b]
    return ("fits " + ", ".join(fits)) if fits else "wider than a typical single voice type"

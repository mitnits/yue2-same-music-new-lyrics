"""Voice repairs for YuE2-native ABC (two voices: Vocal carries chord symbols, Ins is the instrumental melody).

SheetSage2 sometimes files the sung melody under `Ins` (or only in some sections). These helpers move material
between the voices at bar granularity and re-validate the result with abc_tools.

Approximation: chord symbols inside a bar are re-attached at the START of that bar when a bar's notes are replaced,
so a mid-bar chord change moves to the barline. The original is never modified; always keep it.
"""
from __future__ import annotations

import re

from . import abc_tools

CHORD = re.compile(r'"[^"]*"')
NOTE = re.compile(r"[_^=]*[A-Ga-g][,']*")
ALLOWED = [48, 32, 24, 16, 12, 8, 6, 4, 3, 2, 1]


def _rest(units: int) -> str:
    """A rest spanning `units` of the unit length, as allowed duration tokens."""
    out, left = [], units
    for a in ALLOWED:
        while left >= a:
            out.append(f"z{a}")
            left -= a
    return "".join(out) or "z"


def _units_per_bar(meter: str, unit_den: int) -> int:
    n, d = (int(x) for x in meter.split("/"))
    return n * unit_den // d


def _expand(line: str):
    bars = []
    for bar in line[:-1].split("|"):
        bar = bar.strip()
        m = re.fullmatch(r"Z([2-4])?", bar)
        bars.extend(["Z"] * int(m.group(1) or "1") if m else [bar])
    return bars


def _has_notes(bar: str) -> bool:
    return bool(NOTE.search(CHORD.sub("", bar)))


def _notes_only(bar: str) -> str:
    return CHORD.sub("", bar)


def _with_chords(chords: list[str], bar: str, units: int) -> str:
    """Prepend chord symbols to a bar; a compressed rest gets an explicit rest so a chord can attach to it."""
    if bar == "Z" and chords:
        bar = _rest(units)
    return "".join(chords) + bar


def transform(abc: str, mode: str) -> tuple[str, list[str]]:
    """mode: 'swap' | 'merge_ins_into_vocal' | 'vocal_only' | 'ins_as_vocal'. Returns (new_abc, notes)."""
    lines = abc.splitlines()
    if len(lines) < 12 or not lines[3].startswith("L:1/"):
        raise ValueError("Not a YuE2-native two-voice ABC score")
    unit_den = int(lines[3][4:])
    meter = {"Vocal": lines[2][2:].strip(), "Ins": lines[2][2:].strip()}
    out = lines[:8]
    i, changed, notes = 8, 0, []
    while i < len(lines):
        if lines[i].startswith("% ") or not lines[i].strip():
            out.append(lines[i]); i += 1; continue
        # one group: V: Vocal [M:/K:]* music, V: Ins [M:/K:]* music
        parts = {}
        for voice in ("Vocal", "Ins"):
            if i >= len(lines) or lines[i] != f"V: {voice}":
                raise ValueError(f"Unexpected line {i + 1}: expected 'V: {voice}'")
            i += 1
            fields = []
            while i < len(lines) and lines[i].startswith(("M:", "K:")):
                if lines[i].startswith("M:"):
                    meter[voice] = lines[i][2:].strip()
                fields.append(lines[i]); i += 1
            parts[voice] = (fields, _expand(lines[i])); i += 1
        vf, vb = parts["Vocal"]; inf, ib = parts["Ins"]
        if len(vb) != len(ib):
            raise ValueError("Voices have different measure counts in a group")
        units = _units_per_bar(meter["Vocal"], unit_den)
        new_v, new_i = [], []
        for v, s in zip(vb, ib):
            chords = CHORD.findall(v)
            if mode == "swap":
                new_v.append(_with_chords(chords, _notes_only(s) if s != "Z" else "Z", units))
                new_i.append(_notes_only(v) if _has_notes(v) else "Z")
                changed += 1
            elif mode == "merge_ins_into_vocal":
                if not _has_notes(v) and _has_notes(s):
                    new_v.append(_with_chords(chords, _notes_only(s), units)); new_i.append("Z"); changed += 1
                else:
                    new_v.append(v); new_i.append(s)
            elif mode == "vocal_only":
                new_v.append(v); new_i.append("Z"); changed += int(_has_notes(s))
            elif mode == "ins_as_vocal":
                new_v.append(_with_chords(chords, _notes_only(s) if s != "Z" else "Z", units)); new_i.append("Z"); changed += 1
            else:
                raise ValueError(f"Unknown mode {mode}")
        out += ["V: Vocal", *vf, "|".join(new_v) + "|", "V: Ins", *inf, "|".join(new_i) + "|"]
    new = "\n".join(out) + ("\n" if abc.endswith("\n") else "")
    abc_tools.parse_abc(new)  # fail loudly rather than hand YuE2 a broken score
    notes.append(f"{mode}: {changed} bars changed; chord symbols inside replaced bars were moved to the barline.")
    return new, notes


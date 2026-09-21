"""Lyric aligner: a note-level model of the Vocal voice with edit operations that need no music notation.

The model keeps every non-Vocal line of the ABC verbatim and rebuilds the Vocal lines from events, so anything the
user does here round-trips through abc_tools.parse_abc. Sung notes = tie chains merged, exactly as the fit check
counts them. Operations are expressed for non-musicians:

    split   : one note -> two notes of the same pitch (room for one more syllable)
    merge   : this note absorbs the next one (one syllable fewer)
    longer  : this note takes `step` units from what follows in the bar (a shorter next note or rest)
    shorter : this note gives `step` units to a rest after it (created if needed)
    up/down : move the pitch one scale step in the key (stays inside the key, so it stays "in tune")
    sing    : turn a rest into a note (pitch of the previous note) for a pickup syllable
    silence : turn a note into a rest

All durations are integers in the score's unit length (L:1/N). Arbitrary integers are serialized as tied chains of
allowed ABC durations, so any edit is representable.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from . import abc_tools
from .abc_tools import NATURAL, TOKEN, key_accidentals
from .transpose import _emit_note, _spell, _key_count, midi_name
from . import fit

ALLOWED = [48, 32, 24, 16, 12, 8, 6, 4, 3, 2, 1]
SUNG_MIN_NOTES = 4  # sections with fewer sung notes are pickups/tails (same rule as the fit check)


# ----------------------------------------------------------------------------- model
@dataclass
class Event:
    kind: str                      # "note" | "rest"
    dur: int                       # units
    pitch: int | None = None       # MIDI
    tie: bool = False              # tied from the previous event (continuation of the same sung note)
    chords: list = field(default_factory=list)  # chord symbols written just before this event
    compressed: bool = False       # a "Z" whole-bar rest


@dataclass
class Bar:
    events: list                   # list[Event]
    units: int                     # bar length in units
    key: str                       # key in force
    section: str                   # section label ("verse 2")
    group: int                     # index of the V: group this bar belongs to
    index: int                     # bar number in the Vocal voice (0-based)


@dataclass
class Model:
    lines: list                    # original ABC lines
    unit_den: int
    header_key: str
    bars: list                     # list[Bar]
    groups: list                   # per group: dict(vocal_line_idx, fields=[...], bar_range=(a,b))


def _units_per_bar(meter: str, unit_den: int) -> int:
    n, d = (int(x) for x in meter.split("/"))
    return n * unit_den // d


def parse(abc: str) -> Model:
    lines = abc.splitlines()
    if len(lines) < 12 or not lines[3].startswith("L:1/") or not lines[7].startswith("K:"):
        raise ValueError("Not a YuE2-native two-voice ABC score")
    unit_den = int(lines[3][4:])
    meter = lines[2][2:].strip()
    key = lines[7][2:].strip()
    bars, groups = [], []
    section, seen = "song", {}
    i, g = 8, 0
    v_meter, v_key = meter, key
    pending = [None]  # open tie carried across music lines
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("% "):
            base = ln[2:].strip()
            seen[base] = seen.get(base, 0) + 1
            section = base if seen[base] == 1 else f"{base} {seen[base]}"
            i += 1
            continue
        if not ln.strip():
            i += 1
            continue
        if ln != "V: Vocal":
            raise ValueError(f"line {i + 1}: expected 'V: Vocal'")
        i += 1
        fields = []
        while lines[i].startswith(("M:", "K:")):
            if lines[i].startswith("M:"):
                v_meter = lines[i][2:].strip()
            else:
                v_key = lines[i][2:].strip()
            fields.append(lines[i]); i += 1
        vocal_idx = i
        start = len(bars)
        bars.extend(_parse_line(lines[i], v_meter, v_key, unit_den, section, g, start, pending))
        i += 1
        if lines[i] != "V: Ins":
            raise ValueError(f"line {i + 1}: expected 'V: Ins'")
        i += 1
        while lines[i].startswith(("M:", "K:")):
            i += 1
        i += 1  # Ins music line
        groups.append({"vocal_line_idx": vocal_idx, "fields": fields, "bar_range": (start, len(bars))})
        g += 1
    return Model(lines, unit_den, key, bars, groups)


def _parse_line(line: str, meter: str, key: str, unit_den: int, section: str, group: int, start: int,
                pending: list) -> list:
    units = _units_per_bar(meter, unit_den)
    alt = key_accidentals(key)
    out = []
    pending_written = pending[0]
    for raw in line[:-1].split("|"):
        raw = raw.strip()
        m = re.fullmatch(r"Z([2-4])?", raw)
        if m:
            for _ in range(int(m.group(1) or "1")):
                out.append(Bar([Event("rest", units, compressed=True)], units, key, section, group, start + len(out)))
            pending_written = None
            continue
        local = {}
        events, chords, cursor = [], [], 0
        while cursor < len(raw):
            if raw[cursor].isspace():
                cursor += 1; continue
            t = TOKEN.match(raw, cursor)
            if t is None:
                raise ValueError(f"unsupported token {raw[cursor:cursor + 12]!r}")
            cursor = t.end()
            if t.group("chord") is not None:
                chords.append(t.group("chord")); continue
            if t.group("key") is not None:
                key = t.group("key"); alt = key_accidentals(key); local = {}; continue
            note, acc, octave, dur, tie = t.group("note", "acc", "oct", "duration", "tie")
            d = int(dur or "1")
            if note == "z":
                events.append(Event("rest", d, chords=chords)); chords = []; pending_written = None; continue
            letter = note.upper()
            written = 60 + NATURAL[letter] + (12 if note.islower() else 0) + 12 * (octave.count("'") - octave.count(","))
            alteration = local.get(letter, alt[letter])
            if acc:
                alteration = {"=": 0, "_": -1, "__": -2, "^": 1, "^^": 2}[acc]; local[letter] = alteration
            pitch = written + alteration
            continuation = pending_written is not None and not acc and written == pending_written[1]
            if continuation:
                pitch = pending_written[0]
            events.append(Event("note", d, pitch, tie=continuation, chords=chords)); chords = []
            pending_written = (pitch, written) if tie else None
        if chords:  # chord symbols at the very end of a bar: attach to a zero-length marker on the next event
            events[-1].chords.extend(chords)
        out.append(Bar(events, units, key, section, group, start + len(out)))
    pending[0] = pending_written
    return out


# ----------------------------------------------------------------------------- serialization
def _chunks(d: int) -> list:
    out, left = [], d
    for a in ALLOWED:
        while left >= a:
            out.append(a); left -= a
    return out


def serialize(model: Model) -> str:
    lines = list(model.lines)
    prefer_sharps = _key_count(model.header_key) >= 0
    for grp in model.groups:
        a, b = grp["bar_range"]
        bars_txt = []
        pending = None  # (pitch, letter, alt) of an open tie
        for bar in model.bars[a:b]:
            if len(bar.events) == 1 and bar.events[0].kind == "rest" and bar.events[0].compressed and not bar.events[0].chords:
                bars_txt.append("Z"); pending = None; continue
            alt = key_accidentals(bar.key)
            local, toks = {}, []
            for ev in bar.events:
                toks.extend(f'"{c}"' for c in ev.chords)
                if ev.kind == "rest":
                    toks.extend(f"z{c}" for c in _chunks(ev.dur)); pending = None; continue
                if ev.tie and pending and pending[0] == ev.pitch:
                    letter, a_ = pending[1], pending[2]
                else:
                    letter, a_ = _spell(ev.pitch, alt, prefer_sharps)
                chunks = _chunks(ev.dur)
                for k, c in enumerate(chunks):
                    tie = "-" if k < len(chunks) - 1 else ""
                    toks.append(_emit_note(ev.pitch, a_, letter, tie, str(c), local, alt))
                pending = (ev.pitch, letter, a_)
            # a note followed by a tied continuation in the NEXT event needs a trailing "-": handled by look-ahead below
            bars_txt.append(_fix_ties(toks, bar))
        lines[grp["vocal_line_idx"]] = "|".join(bars_txt) + "|"
    text = "\n".join(lines) + ("\n" if "\n".join(model.lines).endswith("\n") or True else "")
    abc_tools.parse_abc(text)
    return text


def _fix_ties(toks: list, bar: Bar) -> str:
    """Add '-' to the last chunk of a note whose following event is a tie continuation (may be in the next bar)."""
    # rebuild positions: map event -> token index of its last note chunk
    out = list(toks)
    ti = 0
    last_chunk_of = {}
    for ei, ev in enumerate(bar.events):
        ti += len(ev.chords)
        n = len(_chunks(ev.dur))
        last_chunk_of[ei] = ti + n - 1
        ti += n
    for ei, ev in enumerate(bar.events):
        nxt = bar.events[ei + 1] if ei + 1 < len(bar.events) else None
        if ev.kind == "note" and nxt is not None and nxt.kind == "note" and nxt.tie:
            k = last_chunk_of[ei]
            if not out[k].endswith("-"):
                out[k] += "-"
    # tie into the next bar is marked by the caller via bar.events[-1]._tie_out (set in serialize_model)
    if getattr(bar, "_tie_out", False) and bar.events and bar.events[-1].kind == "note":
        k = last_chunk_of[len(bar.events) - 1]
        if not out[k].endswith("-"):
            out[k] += "-"
    return "".join(out)


def to_abc(model: Model) -> str:
    # mark cross-bar ties
    for i, bar in enumerate(model.bars):
        nxt = model.bars[i + 1] if i + 1 < len(model.bars) else None
        bar._tie_out = bool(nxt and nxt.events and nxt.events[0].kind == "note" and nxt.events[0].tie
                            and bar.events and bar.events[-1].kind == "note")
    return serialize(model)


# ----------------------------------------------------------------------------- sung-note view
def sung_notes(model: Model) -> list:
    """Sung notes with their first event address; ties merged. Each: dict(id, bar, ev, pitch, dur, start, section)."""
    out, pos = [], 0
    for bar in model.bars:
        off = 0
        for ei, ev in enumerate(bar.events):
            if ev.kind == "note":
                if ev.tie and out and out[-1]["_open"]:
                    out[-1]["dur"] += ev.dur
                else:
                    out.append({"id": len(out), "bar": bar.index, "ev": ei, "pitch": ev.pitch, "dur": ev.dur,
                                "start": pos + off, "section": bar.section, "_open": True})
                out[-1]["_open"] = True
            else:
                if out:
                    out[-1]["_open"] = False
            off += ev.dur
        pos += bar.units
        if out and not (bar.events and bar.events[-1].kind == "note"):
            out[-1]["_open"] = False
    for n in out:
        n.pop("_open", None)
        n["name"] = midi_name(n["pitch"])
    return out


def sections(model: Model) -> list:
    """Sung sections in order, folding tiny pickup sections into the next one (same rule as the fit check)."""
    per = {}
    order = []
    for n in sung_notes(model):
        if n["section"] not in per:
            per[n["section"]] = []; order.append(n["section"])
        per[n["section"]].append(n["id"])
    merged, carry = [], []
    for i, s in enumerate(order):
        ids = per[s]
        if len(ids) <= 3 and len(order) > 1:
            if i + 1 < len(order):
                carry += ids
            elif merged:
                merged[-1]["notes"] += ids
            continue
        merged.append({"name": s, "notes": carry + ids}); carry = []
    return merged


# ----------------------------------------------------------------------------- syllables
def syllabify(word: str, method: str) -> list:
    """Split a word into syllable chips (display + assignment). Rough scripts return one chip weighted by the estimate."""
    w = word.strip()
    if not w:
        return []
    if method == "cjk":
        return [c for c in w if not c.isspace()]
    if method == "cyrillic":
        vowels = set(fit.CYRILLIC_VOWELS)
        idx = [i for i, ch in enumerate(w) if ch.lower() in vowels]
        if len(idx) <= 1:
            return [w]
        cuts = [i + 1 for i in idx[:-1]]  # cut after every vowel except the last (trailing consonants stay)
        pieces, prev = [], 0
        for c in cuts:
            pieces.append(w[prev:c]); prev = c
        pieces.append(w[prev:])
        return pieces
    if method in ("english", "latin"):
        n = fit.count_syllables_word(w, method)
        if n <= 1:
            return [w]
        # split at vowel-group boundaries: keep consonant clusters with the following vowel where possible
        vowels = fit.LATIN_VOWELS if method == "latin" else "aeiouy"
        groups = [m.span() for m in re.finditer(f"[{vowels}]+", w.lower())]
        if len(groups) < n:
            return [w] * 0 + [w] + ["~"] * (n - 1)
        cuts = []
        for (s1, e1), (s2, e2) in zip(groups, groups[1:]):
            mid = e1 + max(1, (s2 - e1) // 2) if s2 - e1 > 1 else s2
            cuts.append(mid)
        pieces, prev = [], 0
        for c in cuts[:n - 1]:
            pieces.append(w[prev:c]); prev = c
        pieces.append(w[prev:])
        return [p for p in pieces if p]
    n = fit.count_syllables_word(w, method)  # hebrew / arabic / indic: rough, one chip per estimated syllable
    return [w] + [f"·{k + 2}" for k in range(max(0, n - 1))]


def lyric_chips(lyrics: str, language: str) -> list:
    """[{tag, lines:[[chip,...],...]}] in the order of the lyrics' [Section] tags (empty/parenthetical lines skipped)."""
    method = fit.method_for(language)
    out = []
    for tag, lines_, _, _ in fit.lyric_sections(lyrics or "", method):
        m = method if method != "auto" else "auto"
        chip_lines = []
        for ln in lines_:
            mm = fit.detect_method(ln) if m == "auto" else m
            chips = []
            for word in re.split(r"[\s\-–—/]+", ln):
                word = word.strip(",.;:!?\"'()…")
                if word:
                    chips.extend(syllabify(word, mm))
            chip_lines.append(chips)
        out.append({"tag": tag, "lines": chip_lines})
    return out


def phrase_gap(model: Model, notes: list, i: int) -> int:
    """Units of rest between sung note i and i+1 of `notes` (0 when they touch)."""
    a, b = notes[i], notes[i + 1]
    return max(0, b["start"] - (a["start"] + a["dur"]))


def auto_starts(notes: list, syllables: list, beat: int) -> list:
    """Line start offsets (line 0 = 0): notes shared out in proportion to syllables, each boundary snapped to the
    nearest rest within ±2 notes. Every line keeps at least one note while notes last."""
    n, m = len(notes), len(syllables)
    if m == 0:
        return []
    total = sum(syllables) or m
    rest_after = {i for i in range(n - 1) if phrase_gap(None, notes, i) >= max(1, beat // 2)}  # boundary after note i
    starts, cum = [0], 0
    for k in range(1, m):
        cum += syllables[k - 1]
        target = round(cum / total * n)
        # snap to a rest if one sits next to the proportional boundary; never move more than one note for it
        cands = [b for b in (target, target - 1, target + 1) if (b - 1) in rest_after]
        b = cands[0] if cands else target
        lo = starts[-1] + 1
        hi = max(lo, n - (m - k))           # leave one note for each remaining line
        starts.append(min(max(b, lo), hi))
    return starts


def runs_from_starts(starts: list, n: int) -> list:
    out = []
    for k, a in enumerate(starts):
        b = starts[k + 1] if k + 1 < len(starts) else n
        a, b = min(a, n), min(max(b, a), n)
        out.append((a, b))
    return out


def align(model: Model, lyrics: str, language: str, line_starts: dict | None = None) -> dict:
    """Section by section: lyric lines matched to phrases of sung notes, syllables assigned inside each line.

    line_starts: {section_index: [absolute_start_units per line]} user reflow (the note whose start is >= that
    position becomes the line's first note); absent = automatic proportional split."""
    notes = sung_notes(model)
    secs = sections(model)
    chips = lyric_chips(lyrics, language)
    beat = max(1, model.unit_den // 4)
    view_sections = []
    for si, sec in enumerate(secs):
        ids = sec["notes"]
        sec_notes = [notes[i] for i in ids]
        lyric = chips[si] if si < len(chips) else None
        lines = lyric["lines"] if lyric else []
        manual = (line_starts or {}).get(str(si))
        if manual and lines:
            offs = []
            for li in range(len(lines)):
                pos = manual[li] if li < len(manual) else None
                off = next((j for j, nn in enumerate(sec_notes) if nn["start"] >= pos), len(sec_notes)) if pos is not None else (offs[-1] + 1 if offs else 0)
                offs.append(off)
            offs[0] = 0
            for j in range(1, len(offs)):  # keep strictly increasing where possible
                offs[j] = max(offs[j], offs[j - 1] + 1) if offs[j - 1] + 1 < len(sec_notes) else min(offs[j], len(sec_notes))
            starts_list = offs
        else:
            starts_list = auto_starts(sec_notes, [len(l) for l in lines], beat)
        runs = runs_from_starts(starts_list, len(sec_notes))
        used = set()
        line_views = []
        for li, (a, b) in enumerate(runs):
            run = sec_notes[a:b]
            syls = lines[li]
            assigned = []
            for k, nn in enumerate(run):
                used.add(nn["id"])
                assigned.append(dict(nn, syllable={"line": li, "text": syls[k]} if k < len(syls) else {"line": None, "text": "~"}))
            line_views.append({"index": li, "chips": syls, "notes": assigned, "note_count": len(run),
                               "syllable_count": len(syls), "overflow": syls[len(run):],
                               "start_pos": run[0]["start"] if run else None, "start_offset": a})
        lead_in = [dict(nn, syllable={"line": None, "text": "~"}) for nn in sec_notes if nn["id"] not in used]
        view_sections.append({"index": si, "name": sec["name"], "tag": lyric["tag"] if lyric else None,
                              "lines": line_views, "lead_in": lead_in, "manual": bool(manual),
                              "starts": starts_list, "start_positions": [sec_notes[o]["start"] if o < len(sec_notes) else None for o in starts_list],
                              "note_count": len(ids), "syllable_count": sum(len(l) for l in lines),
                              "notes": [dict(nn, syllable={"line": None, "text": ""}) for nn in sec_notes] if not lines else []})
    return {"unit_den": model.unit_den, "beat": beat, "key": model.header_key, "sections": view_sections,
            "extra_lyric_sections": [c["tag"] for c in chips[len(secs):]],
            "bars": [{"index": b.index, "units": b.units, "section": b.section} for b in model.bars]}


# ----------------------------------------------------------------------------- edit operations
def _scale(key: str) -> list:
    alt = key_accidentals(key)
    return sorted({(NATURAL[l] + alt[l]) % 12 for l in NATURAL})


def _step_pitch(pitch: int, key: str, direction: int) -> int:
    pcs = _scale(key)
    p = pitch + direction
    while p % 12 not in pcs:
        p += direction
    return p


def apply(abc: str, op: str, note_id: int, step: int | None = None) -> tuple[str, str]:
    """Apply one operation to sung note `note_id`. Returns (new_abc, message)."""
    model = parse(abc)
    notes = sung_notes(model)
    if not 0 <= note_id < len(notes):
        raise ValueError(f"no sung note {note_id}")
    n = notes[note_id]
    bar = model.bars[n["bar"]]
    ev = bar.events[n["ev"]]
    grid = step or min([e.dur for b in model.bars for e in b.events if not e.compressed] or [2])
    if op == "split":
        if ev.dur < 2:
            raise ValueError("this note is already the shortest possible")
        first = max([a for a in ALLOWED if a <= ev.dur // 2] or [1])
        second = ev.dur - first
        bar.events.insert(n["ev"] + 1, Event("note", second, ev.pitch, tie=False, chords=[]))
        ev.dur = first
        msg = f"split note {note_id} ({n['name']}) into two: {first} + {second} units"
    elif op == "merge":
        if note_id + 1 >= len(notes):
            raise ValueError("no following note to merge with")
        m = notes[note_id + 1]
        mbar, mev = model.bars[m["bar"]], model.bars[m["bar"]].events[m["ev"]]
        # anything (rests) between them stays; the next note simply becomes a tied continuation at this pitch
        mev.pitch = ev.pitch; mev.tie = True
        # propagate the pitch through the rest of that note's tie chain
        k = m["ev"] + 1
        while k < len(mbar.events) and mbar.events[k].kind == "note" and mbar.events[k].tie:
            mbar.events[k].pitch = ev.pitch; k += 1
        touching = (m["bar"] == n["bar"] and not _has_rest_between(bar, n["ev"], m["ev"])) or \
                   (m["bar"] == n["bar"] + 1 and n["ev"] == len(bar.events) - 1 and m["ev"] == 0)
        if not touching:
            raise ValueError("can only merge two notes that touch each other (no rest between them)")
        msg = f"merged notes {note_id} and {note_id + 1} into one {n['name']} of {n['dur'] + m['dur']} units"
    elif op == "longer":
        nxt = bar.events[n["ev"] + 1] if n["ev"] + 1 < len(bar.events) else None
        if nxt is None:
            raise ValueError("nothing after this note in its bar to take time from (bar lengths are fixed)")
        if nxt.dur <= grid and nxt.kind == "note":
            raise ValueError("the next note is already as short as it can be")
        take = min(grid, nxt.dur)
        ev.dur += take; nxt.dur -= take
        if nxt.dur == 0:
            bar.events.pop(n["ev"] + 1)
        msg = f"note {note_id} is {take} units longer"
    elif op == "shorter":
        if ev.dur <= grid:
            raise ValueError("this note is already the shortest possible")
        ev.dur -= grid
        nxt = bar.events[n["ev"] + 1] if n["ev"] + 1 < len(bar.events) else None
        if nxt is not None and nxt.kind == "rest":
            nxt.dur += grid
        else:
            bar.events.insert(n["ev"] + 1, Event("rest", grid))
        msg = f"note {note_id} is {grid} units shorter (a short breath follows it)"
    elif op in ("up", "down"):
        newp = _step_pitch(ev.pitch, bar.key, 1 if op == "up" else -1)
        # move the whole tie chain
        for b in model.bars[n["bar"]:]:
            done = False
            for k, e in enumerate(b.events):
                if b is bar and k < n["ev"]:
                    continue
                if e.kind == "note" and (k == n["ev"] and b is bar or e.tie):
                    e.pitch = newp
                else:
                    done = True; break
            if done:
                break
        msg = f"note {note_id}: {n['name']} → {midi_name(newp)}"
    elif op == "silence":
        ev.kind = "rest"; ev.pitch = None; ev.tie = False
        k = n["ev"] + 1
        while k < len(bar.events) and bar.events[k].kind == "note" and bar.events[k].tie:
            bar.events[k].kind = "rest"; bar.events[k].pitch = None; bar.events[k].tie = False; k += 1
        msg = f"note {note_id} is now a rest"
    else:
        raise ValueError(f"unknown operation {op}")
    return to_abc(model), msg


def _has_rest_between(bar: Bar, a: int, b: int) -> bool:
    return any(e.kind == "rest" for e in bar.events[a + 1:b])


def sing_rest(abc: str, bar_index: int, rest_pos: int, units: int | None = None) -> tuple[str, str]:
    """Turn (part of) the rest at event position `rest_pos` in bar `bar_index` into a note at the previous note's pitch.
    `units` limits the note length (default: whole rest, capped at 8 units so a pickup stays short)."""
    model = parse(abc)
    bar = model.bars[bar_index]
    ev = bar.events[rest_pos]
    if ev.kind != "rest":
        raise ValueError("that position is not a rest")
    prev_pitch = None
    for b in reversed(model.bars[:bar_index + 1]):
        evs = b.events[:rest_pos] if b is bar else b.events
        for e in reversed(evs):
            if e.kind == "note":
                prev_pitch = e.pitch; break
        if prev_pitch is not None:
            break
    if prev_pitch is None:  # nothing before: use the first note after
        for b in model.bars[bar_index:]:
            for e in b.events:
                if e.kind == "note":
                    prev_pitch = e.pitch; break
            if prev_pitch is not None:
                break
    if prev_pitch is None:
        raise ValueError("the score has no notes to copy a pitch from")
    length = min(units or 8, ev.dur)
    chords = ev.chords
    if ev.compressed:
        ev.compressed = False
    if length == ev.dur:
        ev.kind = "note"; ev.pitch = prev_pitch; ev.tie = False
    else:
        # note takes the END of the rest (a pickup right before what follows)
        ev.dur -= length
        bar.events.insert(rest_pos + 1, Event("note", length, prev_pitch))
        ev.chords = chords
    return to_abc(model), f"new {midi_name(prev_pitch)} note of {length} units in bar {bar_index + 1}"


def rests_view(model: Model) -> list:
    """Rest positions the editor can offer as 'add a note here': [{bar, pos, dur, start}]."""
    out, pos = [], 0
    for bar in model.bars:
        off = 0
        for ei, ev in enumerate(bar.events):
            if ev.kind == "rest":
                out.append({"bar": bar.index, "pos": ei, "dur": ev.dur, "start": pos + off, "section": bar.section})
            off += ev.dur
        pos += bar.units
    return out

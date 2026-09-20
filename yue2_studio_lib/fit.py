"""Lyrics-fit and ABC helpers for the ComfyUI pack.

Language handling
-----------------
What is known about YuE2's languages (checked 2026-09-20):
  * model card (huggingface.co/m-a-p/YuE2-3B): language: zh, en
  * ComfyUI tutorial (docs.comfy.org): "English, Chinese, and Japanese"
  * official demo page data (map-yue2.github.io/data/cases.js): songs tagged en 56, zh 31, ja 9, ru 1, ko 1, es 1
  * MindStudio hands-on tests: Russian, Bengali, Brazilian Portuguese, French, German, Swedish, Indonesian, Dutch,
    Tagalog, Persian, Polish, Arabic "passable to good"; Hindi and Urdu "notably rough"
  * no published evidence for Hebrew
The language tag is the FIRST tag of the style prompt ("English, ..."), and the lyrics themselves carry the
language; there is no separate language input in the model.

Syllable counting is per language/script, and every report states the method used:
  english            vowel groups with silent-e / -ed rules (about ±1 per line)
  latin              plain vowel groups (Spanish, Italian, Portuguese, German, Dutch, Polish, Turkish… ±1 per line)
  cyrillic           one syllable per vowel letter (exact for Russian/Ukrainian)
  cjk                one syllable per Han character / kana (small kana skipped) / hangul block
  hebrew             ROUGH: unvocalized script has no written vowels; ≈0.65 syllables per letter
  arabic             ROUGH: ≈0.6 syllables per letter (short vowels are unwritten)
  auto               detect the script of each line and pick one of the above
  off                no syllable counting; only section/line structure is compared
"""
from __future__ import annotations

import re
from pathlib import Path

from . import abc_tools

NOTE_TOKEN = re.compile(r"(-?)([_^=]*[A-Ga-g][,']*)(\d*)")
CYRILLIC_VOWELS = "аеёиоуыэюяіїєґ"
LATIN_VOWELS = "aeiouyáéíóúàèìòùâêîôûäëïöüãõåæøœąęőű"

# Languages offered in the UIs. The first tag of the style prompt uses the display name.
LANGUAGES = ["English", "Chinese", "Japanese", "Russian", "Korean", "Spanish", "Portuguese", "French", "German",
             "Polish", "Swedish", "Dutch", "Indonesian", "Tagalog", "Persian", "Bengali", "Arabic", "Italian",
             "Ukrainian", "Hindi", "Urdu", "Hebrew", "Other"]
# Evidence tiers (see module docstring)
DOCUMENTED = {"English", "Chinese", "Japanese"}
OFFICIAL_DEMO = {"Russian", "Korean", "Spanish"}
COMMUNITY_GOOD = {"Portuguese", "French", "German", "Swedish", "Indonesian", "Dutch", "Tagalog", "Persian", "Bengali",
                  "Polish", "Arabic"}
COMMUNITY_WEAK = {"Hindi", "Urdu"}
NATIVE_LANGUAGES = DOCUMENTED | OFFICIAL_DEMO
LANGUAGE_METHOD = {"English": "english", "Chinese": "cjk", "Japanese": "cjk", "Korean": "cjk", "Russian": "cyrillic",
                   "Ukrainian": "cyrillic", "Hebrew": "hebrew", "Arabic": "arabic", "Persian": "arabic",
                   "Urdu": "arabic", "Hindi": "indic", "Bengali": "indic", "Spanish": "latin", "Italian": "latin",
                   "Portuguese": "latin", "French": "latin", "German": "latin", "Polish": "latin", "Swedish": "latin",
                   "Dutch": "latin", "Indonesian": "latin", "Tagalog": "latin", "Other": "auto"}
METHODS = ["auto", "english", "latin", "cyrillic", "cjk", "hebrew", "arabic", "indic", "off"]


def language_note(language: str) -> str:
    """One-line caveat for the UI when the chosen language is outside YuE2's documented set."""
    if not language or language in DOCUMENTED:
        return ""
    if language in OFFICIAL_DEMO:
        return f"ℹ {language}: not on the model card, but the official YuE2 demo page showcases a {language} song."
    if language in COMMUNITY_GOOD:
        return (f"ℹ {language}: not documented; community hands-on tests report passable to good vocals. "
                "Listen to a short test before writing a full song.")
    if language in COMMUNITY_WEAK:
        return f"⚠ {language}: community tests report noticeably rough vocals."
    return (f"⚠ {language}: no published evidence that YuE2 can sing it (documented: English, Chinese, Japanese; "
            "demoed: Russian, Korean, Spanish). Best effort only; run a short test first.")


# ----------------------------------------------------------------------------- score side
def vocal_notes_per_section(abc: str):
    """Ordered list of (section, sounding vocal notes). Ties (`-`) merge into the previous note."""
    return [(s, sum(p)) for s, p in vocal_phrases_per_section(abc, fold_pickups=False)]


def vocal_phrases_per_section(abc: str, fold_pickups: bool = True):
    """Ordered list of (section, [sounding vocal notes per music line]); a music line is 1-4 bars.
    With fold_pickups, a section holding <=3 notes (a pickup or the tail of the previous word) is folded into
    the next sung section (or the previous one at the end)."""
    out, section, voice, seen = [], "song", None, {}
    lines = abc.splitlines()
    for ln in lines[8:] if len(lines) > 8 else []:
        if ln.startswith("% "):
            base = ln[2:].strip()
            seen[base] = seen.get(base, 0) + 1
            section = base if seen[base] == 1 else f"{base} {seen[base]}"
            out.append((section, []))
            continue
        if ln.startswith("V: "):
            voice = ln[3:].strip()
            continue
        if ln.startswith(("M:", "K:")) or voice != "Vocal":
            continue
        if not out:
            out.append((section, []))
        body = re.sub(r'"[^"]*"', "", ln).replace("|", "")  # drop chords and barlines so cross-bar ties merge
        out[-1][1].append(sum(1 for tie, _, _ in NOTE_TOKEN.findall(body) if tie != "-"))
    sung = [(s, [n for n in phrases if n > 0]) for s, phrases in out if sum(phrases) > 0]
    if not fold_pickups:
        return sung
    merged, carry = [], 0
    for i, (name, phrases) in enumerate(sung):
        if sum(phrases) <= 3 and len(sung) > 1:
            if i + 1 < len(sung):
                carry += sum(phrases)
            elif merged:
                merged[-1][1][-1] += sum(phrases)
            continue
        if carry:
            phrases = [phrases[0] + carry] + phrases[1:]
            carry = 0
        merged.append((name, list(phrases)))
    return merged


def score_facts(abc: str, structure_lab: str | None = None) -> dict:
    """Measured facts for prompts/describer: tempo, key, meter from the score; section timeline from structure.lab."""
    facts = {}
    if abc:
        m = re.search(r"^Q:1/4=(\d+)", abc, re.M)
        k = re.search(r"^K:(\S+)", abc, re.M)
        t = re.search(r"^M:(\S+)", abc, re.M)
        if m:
            facts["tempo"] = f"{m.group(1)} BPM"
        if k:
            facts["key"] = k.group(1)
        if t:
            facts["meter"] = t.group(1)
    if structure_lab and Path(structure_lab).is_file():
        parts = []
        for ln in Path(structure_lab).read_text().splitlines():
            try:
                a, b, name = ln.split()
                parts.append(f"{name} {float(a):.0f}-{float(b):.0f}s")
            except ValueError:
                continue
        if parts:
            facts["structure"] = "; ".join(parts)
    return facts


# ----------------------------------------------------------------------------- lyrics side
def detect_method(text: str) -> str:
    """Pick a counting method from the dominant script of the text."""
    counts = {
        "cyrillic": len(re.findall(r"[Ѐ-ӿ]", text)),
        "hebrew": len(re.findall(r"[֐-׿]", text)),
        "arabic": len(re.findall(r"[؀-ۿ]", text)),
        "cjk": len(re.findall(r"[぀-ヿ㐀-鿿가-힯]", text)),
        "indic": len(re.findall(r"[\u0900-\u097f\u0980-\u09ff]", text)),
        "latin": len(re.findall(r"[A-Za-zÀ-ɏ]", text)),
    }
    best = max(counts, key=counts.get)
    if counts[best] == 0:
        return "off"
    if best == "latin":
        # English vs other Latin-script languages: cheap function-word test
        words = set(re.findall(r"[a-z']+", text.lower()))
        english = {"the", "and", "you", "i'm", "we", "will", "not", "but", "with", "of", "is", "are", "my", "your"}
        return "english" if len(words & english) >= 2 else "latin"
    return best


def _syllables_english(word: str) -> int:
    w = re.sub(r"[^a-z']", "", word.lower())
    if not w or not re.search(r"[a-z]", w):
        return 0
    n = len(re.findall(r"[aeiouy]+", w))
    if w.endswith("e") and not w.endswith(("le", "ee", "ye", "oe")) and n > 1:
        n -= 1
    if w.endswith("ed") and not w.endswith(("ted", "ded")) and n > 1:
        n -= 1
    return max(1, n)


def _syllables_latin(word: str) -> int:
    w = word.lower()
    if not re.search(f"[{LATIN_VOWELS}]", w):
        return 1 if re.search(r"[a-z]", w) else 0
    return len(re.findall(f"[{LATIN_VOWELS}]+", w))


def count_syllables_word(word: str, method: str = "auto") -> int:
    if method == "auto":
        method = detect_method(word)
    w = word.strip()
    if not w or method == "off":
        return 0
    if method == "english":
        return _syllables_english(w)
    if method == "latin":
        return _syllables_latin(w)
    if method == "cyrillic":
        return len(re.findall(f"[{CYRILLIC_VOWELS}]", w.lower()))  # vowel-less clitics (в, к, с) add nothing
    if method == "cjk":
        han = len(re.findall(r"[㐀-鿿]", w))
        kana = len(re.findall(r"[ぁ-ゖァ-ヶ]", w)) - len(re.findall(r"[ゃゅょャュョぁぃぅぇぉァィゥェォ]", w))
        hangul = len(re.findall(r"[가-힯]", w))
        return han + max(kana, 0) + hangul
    if method == "hebrew":
        letters = len(re.findall(r"[א-ת]", w))
        return max(1, round(letters * 0.65)) if letters else 0
    if method == "indic":  # Devanagari / Bengali: consonant and vowel letters minus virama-killed consonants
        virama = len(re.findall(r"[\u094d\u09cd]", w))
        letters = len(re.findall(r"[\u0904-\u0939\u0985-\u09b9]", w))
        return max(1, letters - virama) if letters else 0
    if method == "arabic":
        letters = len(re.findall(r"[ء-ي]", w))
        return max(1, round(letters * 0.6)) if letters else 0
    return 0


def line_syllables(text: str, method: str = "auto") -> int:
    m = detect_method(text) if method == "auto" else method
    if m == "off":
        return 0
    if m == "cjk":
        return count_syllables_word(text, m)
    return sum(count_syllables_word(w, m) for w in re.split(r"[\s\-–—/]+", text))  # rock-n-roll = 3


def lyric_sections(lyrics: str, method: str = "auto"):
    """Ordered list of (tag, lines, syllables, per_line). Lines that are only "(a note)" are skipped."""
    out, tag, buf = [], "untagged", []

    def flush():
        sung = [ln for ln in buf if not re.fullmatch(r"\(.*\)", ln.strip())]
        if sung:
            per = [line_syllables(ln, method) for ln in sung]
            out.append((tag, list(sung), sum(per), per))

    for ln in lyrics.splitlines():
        m = re.fullmatch(r"\s*\[([^\]]+)\]\s*", ln)
        if m:
            flush()
            tag, buf = m.group(1).strip(), []
        elif ln.strip():
            buf.append(ln.strip())
    flush()
    return out


def method_for(language: str | None, method: str | None = None) -> str:
    if method and method != "auto":
        return method
    if language and language in LANGUAGE_METHOD:
        return LANGUAGE_METHOD[language]
    return "auto"


def fit_report(abc: str, original_lyrics: str, new_lyrics: str, language: str | None = None,
               method: str | None = None) -> str:
    """Markdown table: sung sections of the score (notes per 4-bar phrase) vs the lyrics, section by section.
    `language` (display name) or `method` selects the syllable counter for the NEW lyrics; the original lyrics are
    always counted by script auto-detection. With method 'off', only section and line structure is compared."""
    if not abc or not abc.strip():
        return "_Load a source score first._"
    m_new = method_for(language, method)
    sung = vocal_phrases_per_section(abc)
    old = lyric_sections(original_lyrics or "", "auto")
    new = lyric_sections(new_lyrics or "", m_new)
    counting = m_new != "off"
    rows = ["| # | Score section | Sung notes (per phrase) | Original (syllables) | New lyrics "
            + ("(syllables per line)" if counting else "(lines)") + " | Fit |", "|---|---|---|---|---|---|"]
    n = max(len(sung), len(old), len(new))
    for i in range(n):
        s_name, phrases = sung[i] if i < len(sung) else ("—", None)
        notes = sum(phrases) if phrases else None
        notes_txt = f"**{notes}** ({' / '.join(map(str, phrases))})" if phrases else "—"
        o = f"[{old[i][0]}] {old[i][2]}" if i < len(old) else "—"
        if i < len(new):
            nw = (f"[{new[i][0]}] **{new[i][2]}** ({' / '.join(map(str, new[i][3]))})" if counting
                  else f"[{new[i][0]}] {len(new[i][1])} lines")
        else:
            nw = "—"
        fit = ""
        if notes is not None and i < len(new) and counting:
            ref, ref_name = (old[i][2], "the original") if i < len(old) and old[i][2] else (notes, "the note count")
            diff = new[i][2] - ref
            if abs(diff) <= max(2, ref * 0.1):
                fit = "✅ good"
            elif diff > 0:
                fit = f"⚠ {diff} syllables more than {ref_name} (two syllables share a note in places; fine if the original sang it that way)"
            else:
                fit = f"⚠ {-diff} fewer syllables than {ref_name} (expect stretched vowels)"
        elif notes is not None and i < len(new):
            fit = "section present"
        elif notes is None and i < len(new):
            fit = "⚠ no matching sung section in score"
        elif notes is not None and i >= len(new):
            fit = "⚠ score section has no lyrics"
        rows.append(f"| {i + 1} | {s_name} | {notes_txt} | {o} | {nw} | {fit} |")
    total = f"\n**Total sung notes:** {sum(sum(p) for _, p in sung)}"
    if counting:
        total += f" · **original syllables:** {sum(x[2] for x in old)} · **new syllables:** {sum(x[2] for x in new)}"
    if len(sung) != len(new):
        total += (f"\n\n⚠ The score has **{len(sung)}** sung sections but the lyrics have **{len(new)}** sections "
                  f"with words. Keep the same section order and count as the original. Tags with no words under "
                  f"them (e.g. `[Guitar Solo]`) are fine and ignored here.")
    precision = {"english": "English heuristic, about ±1 per line", "latin": "vowel groups, about ±1 per line",
                 "cyrillic": "one syllable per vowel letter (exact)", "cjk": "one syllable per character",
                 "hebrew": "ROUGH estimate: Hebrew script has no written vowels (≈0.65 per letter)",
                 "arabic": "ROUGH estimate (≈0.6 per letter)", "indic": "letters minus virama (approximate)", "auto": "script auto-detected per line",
                 "off": "no syllable counting"}
    hint = (f"\n\n_Counting method for new lyrics: **{m_new}** ({precision.get(m_new, '')}). "
            "Note counts are a floor: transcriptions snap to an eighth-note grid, so fast syllable pairs merge into "
            "one note. A phrase is one 4-bar music line; keep the section order and put stressed syllables on long notes._")
    ln = language_note(language) if language else ""
    return "\n".join(rows) + total + hint + (f"\n\n{ln}" if ln else "")


def compare_scores(source_abc: str, result_abc: str) -> str:
    try:
        rep = abc_tools.compare(abc_tools.parse_abc(source_abc), abc_tools.parse_abc(result_abc))
        if rep.get("match"):
            return "✅ Score check: melody, rhythm, meter, tempo and chords of the new song are identical to the source score."
        return "⚠ Score check found differences: " + "; ".join(rep.get("differences", []))
    except Exception as exc:
        return f"Score check skipped ({exc})"


def style_with_language(style: str, language: str) -> str:
    """Make `language` the first tag of a style prompt, replacing an existing language tag if present."""
    if not language:
        return style
    parts = [p.strip() for p in (style or "").split(",") if p.strip()]
    if parts and parts[0] in LANGUAGES:
        parts = parts[1:]
    return ", ".join([language] + parts)

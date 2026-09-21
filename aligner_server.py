"""HTTP side of the Lyric Aligner: session files + JSON routes on ComfyUI's own server (prefix /yue2sml)."""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import folder_paths

from .lib import abc_tools, align, fit

HERE = Path(__file__).resolve().parent
OUT_SUBDIR = "yue2-same-music-new-lyrics/aligner"


def _session_dir(name: str) -> Path:
    safe = "".join(c if c.isalnum() or c in "-_." else "-" for c in name) or "session"
    d = Path(folder_paths.get_output_directory()) / OUT_SUBDIR / safe
    d.mkdir(parents=True, exist_ok=True)
    return d


def _state_path(name: str) -> Path:
    return _session_dir(name) / "state.json"


def load_state(name: str) -> dict | None:
    p = _state_path(name)
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


def save_state(name: str, state: dict) -> None:
    _state_path(name).write_text(json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")
    if state.get("edited_abc"):
        (_session_dir(name) / "edited_score.abc").write_text(state["edited_abc"], encoding="utf-8")


def _hash(text: str) -> str:
    return hashlib.sha256((text or "").strip().encode("utf-8")).hexdigest()[:16]


def bar_times(folder: str) -> list:
    """Bar start times in seconds from SheetSage2's beat grid (beat 1 rows), if the transcription folder is known."""
    if not folder:
        return []
    for cand in (Path(folder) / "sheetsage" / "notation" / "song_beats.txt", Path(folder) / "notation" / "song_beats.txt"):
        if cand.is_file():
            times = []
            for ln in cand.read_text().splitlines():
                parts = ln.split()
                if len(parts) >= 2 and parts[1] == "1":
                    times.append(float(parts[0]))
            return times
    return []


def audio_path(folder: str) -> Path | None:
    if not folder:
        return None
    for cand in (Path(folder) / "audio.flac", Path(folder) / "vocal_forward_mix.flac"):
        if cand.is_file():
            return cand
    return None


def prepare_session(name: str, abc: str, lyrics: str, language: str, folder: str = "") -> dict:
    """Called by the node on every run. A new base score invalidates old edits."""
    state = load_state(name) or {}
    h = _hash(abc)
    if state.get("base_hash") != h:
        if state.get("edited_abc"):
            logging.warning(f"[yue2-sml] Lyric Aligner '{name}': the incoming score changed; previous edits set aside "
                            f"(kept as edited_score.previous.abc)")
            (_session_dir(name) / "edited_score.previous.abc").write_text(state["edited_abc"], encoding="utf-8")
        state = {"base_abc": abc, "base_hash": h, "edited_abc": None, "history": [], "line_starts": {}}
    state.update({"lyrics": lyrics, "language": language, "folder": folder, "bar_times": bar_times(folder),
                  "bpm": int((abc_tools.parse_abc(abc)).bpm)})
    save_state(name, state)
    return state


def summary(abc: str, lyrics: str, language: str, edited: bool) -> str:
    model = align.parse(abc)
    view = align.align(model, lyrics, language)
    parts = []
    for s in view["sections"]:
        tag = f"[{s['tag']}]" if s["tag"] else "(no lyrics)"
        bad = sum(1 for L in s["lines"] if L["overflow"] or L["note_count"] - L["syllable_count"] > 2)
        flag = "ok" if s["lines"] and not bad else (f"{bad} line(s) to fix" if s["lines"] else "no lyrics")
        parts.append(f"{s['name']} {s['note_count']} notes ↔ {tag} {s['syllable_count']} syllables ({flag})")
    return ("EDITED score" if edited else "transcribed score") + " · " + "; ".join(parts)


def _kind_at(section: dict, event_index: int) -> str:
    events = [e for L in section["lines"] for e in L["events"]]
    events.sort(key=lambda e: e["start"])
    return events[event_index]["kind"] if 0 <= event_index < len(events) else "note"


def view_payload(name: str, abc: str) -> dict:
    state = load_state(name) or {}
    model = align.parse(abc)
    v = align.align(model, state.get("lyrics", ""), state.get("language", "English"), state.get("line_starts") or {})
    v["rests"] = align.rests_view(model)
    v["bpm"] = model and int(abc_tools.parse_abc(abc).bpm)
    v["bar_times"] = state.get("bar_times", [])
    v["has_audio"] = audio_path(state.get("folder", "")) is not None
    v["edited"] = bool(state.get("edited_abc"))
    v["language"] = state.get("language", "English")
    v["lyrics"] = state.get("lyrics", "")
    v["name"] = name
    return v


# ----------------------------------------------------------------------------- routes
def register_routes():
    try:
        from server import PromptServer
        from aiohttp import web
    except Exception as exc:  # pragma: no cover
        logging.warning(f"[yue2-sml] aligner routes not registered: {exc}")
        return
    routes = PromptServer.instance.routes

    @routes.get("/yue2sml/aligner")
    async def aligner_page(request):
        html = (HERE / "web" / "aligner.html").read_text(encoding="utf-8")
        return web.Response(text=html, content_type="text/html")

    @routes.get("/yue2sml/session")
    async def session(request):
        name = request.query.get("name", "")
        state = load_state(name)
        if not state:
            return web.json_response({"error": f"no aligner session '{name}': run the Lyric Aligner node first"}, status=404)
        abc = state.get("edited_abc") or state["base_abc"]
        payload = view_payload(name, abc)
        payload["abc"] = abc
        return web.json_response(payload)

    @routes.post("/yue2sml/op")
    async def op(request):
        body = await request.json()
        name, abc = body["name"], body["abc"]
        try:
            if body["op"] == "sing":
                new, msg = align.sing_rest(abc, int(body["bar"]), int(body["pos"]), body.get("units"))
            else:
                new, msg = align.apply(abc, body["op"], int(body["note_id"]), body.get("step"))
        except ValueError as exc:
            return web.json_response({"error": str(exc)}, status=400)
        state = load_state(name) or {}
        state["edited_abc"] = new
        save_state(name, state)
        payload = view_payload(name, new)
        payload.update({"abc": new, "msg": msg})
        return web.json_response(payload)

    @routes.post("/yue2sml/set")
    async def set_abc(request):
        """Undo / reset: the client sends the ABC it wants to be current (validated)."""
        body = await request.json()
        name = body["name"]
        state = load_state(name) or {}
        abc = body.get("abc")
        if abc is None or abc.strip() == (state.get("base_abc") or "").strip():
            state["edited_abc"] = None
            abc = state["base_abc"]
            p = _session_dir(name) / "edited_score.abc"
            if p.is_file():
                p.unlink()
        else:
            try:
                abc_tools.parse_abc(abc)
            except Exception as exc:
                return web.json_response({"error": f"invalid score: {exc}"}, status=400)
            state["edited_abc"] = abc
        save_state(name, state)
        payload = view_payload(name, abc)
        payload["abc"] = abc
        return web.json_response(payload)

    @routes.post("/yue2sml/lyrics")
    async def set_lyrics(request):
        body = await request.json()
        name = body["name"]
        state = load_state(name) or {}
        state["lyrics"] = body.get("lyrics", state.get("lyrics", ""))
        state["language"] = body.get("language", state.get("language", "English"))
        save_state(name, state)
        abc = state.get("edited_abc") or state["base_abc"]
        payload = view_payload(name, abc)
        payload["abc"] = abc
        return web.json_response(payload)

    @routes.post("/yue2sml/reflow")
    async def reflow(request):
        """Move one note across a line boundary: {name, section, line, delta} moves the START of `line` by delta notes
        (+1 = give this line's first note to the line above, -1 = take the last note of the line above).
        {name, section, reset: true} returns the section to the automatic split."""
        body = await request.json()
        name = body["name"]
        state = load_state(name) or {}
        abc = state.get("edited_abc") or state["base_abc"]
        ls = state.setdefault("line_starts", {})
        sec = str(int(body["section"]))
        if body.get("reset"):
            ls.pop(sec, None); msg = "line split back to automatic"
        else:
            view = view_payload(name, abc)
            section = view["sections"][int(sec)]
            starts = list(section["starts"])
            n = section["event_count"]
            k, delta = int(body["line"]), int(body["delta"])
            if not 1 <= k < len(starts):
                return web.json_response({"error": "the first line always starts at the first note"}, status=400)
            new = starts[k] + delta
            lo = starts[k - 1] + 1
            hi = (starts[k + 1] - 1) if k + 1 < len(starts) else n - 1
            if new < lo or new > hi:
                return web.json_response({"error": "that would leave a line completely empty (no notes and no pauses)"}, status=400)
            starts[k] = new
            ls[sec] = [section["event_starts"][o] for o in starts]
            moved = "pause" if section["event_starts"] and view["sections"][int(sec)] and _kind_at(view["sections"][int(sec)], starts[k] if delta < 0 else starts[k] - 1) == "rest" else "note"
            msg = (f"gave a {moved} to the line above" if delta > 0 else f"took a {moved} from the line above")
        save_state(name, state)
        payload = view_payload(name, abc)
        payload.update({"abc": abc, "msg": msg})
        return web.json_response(payload)

    @routes.get("/yue2sml/audio")
    async def audio(request):
        state = load_state(request.query.get("name", "")) or {}
        p = audio_path(state.get("folder", ""))
        if p is None:
            return web.Response(status=404, text="no audio for this session")
        return web.FileResponse(p)

    logging.info("[yue2-sml] Lyric Aligner routes registered under /yue2sml")

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
    """Called by the node on every run. A new base score sets note edits aside; new input lyrics replace tool edits."""
    state = load_state(name) or {}
    h = _hash(abc)
    if state.get("base_hash") != h:
        if state.get("edited_abc"):
            logging.warning(f"[yue2-sml] Lyric Aligner '{name}': the incoming score changed; previous edits set aside "
                            f"(kept as edited_score.previous.abc)")
            (_session_dir(name) / "edited_score.previous.abc").write_text(state["edited_abc"], encoding="utf-8")
        state = {"base_abc": abc, "base_hash": h, "edited_abc": None}
    lh = _hash(lyrics)
    if state.get("input_lyrics_hash") != lh:
        state["lyrics"] = lyrics            # the ComfyUI text box changed: it wins over edits made in the tool
        state["input_lyrics_hash"] = lh
        state["lyrics_edited"] = False
    state.update({"language": language, "folder": folder, "bar_times": bar_times(folder),
                  "bpm": int((abc_tools.parse_abc(abc)).bpm)})
    save_state(name, state)
    return state


def summary(abc: str, lyrics: str, language: str, edited: bool) -> str:
    view = align.align(align.parse(abc), lyrics, language)
    parts = []
    for s in view["sections"]:
        tag = f"[{s['tag']}]" if s["tag"] else "(no lyrics)"
        over = sum(len(L["overflow"]) for L in s["lines"])
        held = sum(L["note_count"] for L in s["lines"] if L.get("held"))
        flag = f"{over} syllables without a note" if over else ("ok" if s["lines"] else "no lyrics")
        if held:
            flag += f", {held} notes left over"
        parts.append(f"{s['name']} {s['note_count']} notes ↔ {tag} {s['syllable_count']} syllables ({flag})")
    return ("EDITED score" if edited else "transcribed score") + " · " + "; ".join(parts)


def view_payload(name: str, abc: str) -> dict:
    state = load_state(name) or {}
    model = align.parse(abc)
    v = align.align(model, state.get("lyrics", ""), state.get("language", "English"), state.get("line_extra") or {})
    v["rests"] = align.rests_view(model)
    v["bpm"] = int(abc_tools.parse_abc(abc).bpm)
    v["bar_times"] = state.get("bar_times", [])
    v["has_audio"] = audio_path(state.get("folder", "")) is not None
    v["edited"] = bool(state.get("edited_abc"))
    v["lyrics_edited"] = bool(state.get("lyrics_edited"))
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

    def respond(name, abc, msg=None):
        payload = view_payload(name, abc)
        payload["abc"] = abc
        if msg:
            payload["msg"] = msg
        return web.json_response(payload)

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
        return respond(name, state.get("edited_abc") or state["base_abc"])

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
        return respond(name, new, msg)

    @routes.post("/yue2sml/set")
    async def set_abc(request):
        """Undo / reset: the client sends the ABC it wants to be current (validated); null = back to the transcription."""
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
        return respond(name, abc)

    @routes.post("/yue2sml/lyrics")
    async def set_lyrics(request):
        body = await request.json()
        name = body["name"]
        state = load_state(name) or {}
        if body.get("lyrics") is not None:
            state["lyrics"] = body["lyrics"]; state["lyrics_edited"] = True
        if body.get("language"):
            state["language"] = body["language"]
        save_state(name, state)
        return respond(name, state.get("edited_abc") or state["base_abc"], "lyrics updated (they flow out of the node's lyrics output)")

    @routes.post("/yue2sml/lines")
    async def lines(request):
        """Line edits: {name, section, line, action: insert|delete|set, text?}."""
        body = await request.json()
        name = body["name"]
        state = load_state(name) or {}
        try:
            state["lyrics"] = align.edit_lines(state.get("lyrics", ""), int(body["section"]), int(body["line"]),
                                               body["action"], body.get("text", ""))
        except ValueError as exc:
            return web.json_response({"error": str(exc)}, status=400)
        state["line_extra"] = align.remap_extras(state.get("line_extra") or {}, int(body["section"]), int(body["line"]), body["action"])
        state["lyrics_edited"] = True
        save_state(name, state)
        msg = {"insert": "empty line added", "delete": "line deleted", "set": "line updated"}[body["action"]]
        return respond(name, state.get("edited_abc") or state["base_abc"], msg)

    @routes.post("/yue2sml/extra")
    async def extra(request):
        """{name, section, line, delta}: +1 = this line takes one more note (from the next line), -1 = gives one."""
        body = await request.json()
        name = body["name"]
        state = load_state(name) or {}
        le = state.setdefault("line_extra", {})
        sec, line = str(int(body["section"])), str(int(body["line"]))
        cur = int(le.get(sec, {}).get(line, 0)) + int(body["delta"])
        le.setdefault(sec, {})[line] = cur
        if cur == 0:
            le[sec].pop(line, None)
        save_state(name, state)
        d = int(body["delta"])
        msg = ("took a note from the next line" if d > 0 else "gave a note to the next line") + f" (line now {cur:+d} vs its syllables)"
        return respond(name, state.get("edited_abc") or state["base_abc"], msg)

    @routes.get("/yue2sml/audio")
    async def audio(request):
        state = load_state(request.query.get("name", "")) or {}
        p = audio_path(state.get("folder", ""))
        if p is None:
            return web.Response(status=404, text="no audio for this session")
        return web.FileResponse(p)

    logging.info("[yue2-sml] Lyric Aligner routes registered under /yue2sml")

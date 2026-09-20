# comfyui-yue2-same-music-new-lyrics

Custom nodes that add the "song from anywhere → same melody, new lyrics" workflow pieces around ComfyUI's
built-in YuE2 nodes (CheckpointLoaderSimple + YuE2GenerateABC / YuE2GenerateMusic / EmptyYuE2LatentAudio /
KSampler / VAEDecodeAudio, and AudioEncoderLoader + SheetSage2AudioToABC).

Nodes (category `audio/yue2-same-music-new-lyrics`):

| Node | What it does |
|---|---|
| Transcribe Recording (SheetSage2 full) | Full SheetSage2 release in its own venv: ABC with section comments, chords, key, tempo + timeline, MIDI, annotations. `melody_tracks = vocal only` asks for the sung melody alone |
| Separate Vocals (Demucs) | Vocal stem + accompaniment stem (htdemucs); reports seconds of singing. Transcribe's `isolate_vocals_first` uses it to transcribe a vocal-forward remix so the singer cannot be filed as an instrument |
| Score Editor | The pause between transcribing and rendering: the box fills with the incoming score on first run, you edit it on the canvas and queue again (upstream is cached). Saves the score to `output/yue2_studio/<name>.abc`. `RESET` reloads |
| Transpose | Two range levers: key_shift (whole song, semitones) and vocal_octave (sung line only, octaves); reports the vocal range before/after |
| Fix Voices | Repairs a transcription whose singer landed in the Ins track: merge_ins_into_vocal / swap / vocal_only / ins_as_vocal |
| Describe Track → Style Prompt | Audio Flamingo 3 + CLAP listen to the recording and draft a YuE2 style line (optional, ~17 GB VRAM) |
| Lyrics Fit Check | Syllables per lyric section vs sung notes per phrase; language-aware counting (English, Latin-script, Cyrillic, CJK exact-ish; Hebrew/Arabic rough; or off) |
| Set Song Language | Stamps the language as the first style tag; warns outside YuE2-3B's documented Chinese/English |
| Score Facts | bpm, key, meter, sections, nominal length from an ABC score |
| Strip Chords from Score | Chord-free score for mode `melody` |
| Compare Scores | Exact symbolic melody/rhythm check between two scores |
| Load Audio (file chooser), Load Text File | Dropdown of audio files in ComfyUI/input, ~/Downloads, ~/Music (or any path); text loader |

The heavy nodes call the worker scripts and virtual environments created under `~/yue2-same-music-new-lyrics`
(`sheetsage_worker.py` + `YuE/.venv-sheetsage2`, `describe_worker.py` + `YuE/.venv-describe`).
Point `YUE2_HOME` at that folder if it lives elsewhere.

Models for the native nodes (from https://huggingface.co/Comfy-Org/YuE2):
`models/checkpoints/yue2_3b_bf16.safetensors` (or the int8 variant) and `models/audio_encoders/sheetsage2_bf16.safetensors`.

Example workflows in `workflows/` (also installed in the ComfyUI workflow list):

* `stage1_transcribe_and_edit.json` — Stage 1: recording → score, edit on the canvas, saved to `output/yue2_studio/edited_score.abc`
* `stage2_generate_from_score.json` — Stage 2: load that file (or any `.abc`) → lyrics/language/style → render
* `same_music_new_lyrics.json` — both stages on one canvas with the Score Editor in the middle

`lib/abc_tools.py` is the Apache-2.0 helper from the YuE repository (see `lib/LICENSE.abc_tools`).

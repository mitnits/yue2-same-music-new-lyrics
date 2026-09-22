# Node reference

Custom nodes that add the "song from anywhere → same melody, new lyrics" workflow pieces around ComfyUI's
built-in YuE2 nodes (CheckpointLoaderSimple + YuE2GenerateABC / YuE2GenerateMusic / EmptyYuE2LatentAudio /
KSampler / VAEDecodeAudio, and AudioEncoderLoader + SheetSage2AudioToABC).

Nodes (category `audio/yue2-same-music-new-lyrics`):

| Node | What it does |
|---|---|
| Transcribe Recording (SheetSage2 full) | Full SheetSage2 release in its own venv: ABC with section comments, chords, key, tempo + timeline, MIDI, annotations |
| Lyric Aligner | Note tweaker for non-musicians, opened as a full-window overlay from the node: syllables pour over each section's notes in order, one row per lyric line. Click a note → Split / Merge / Longer / Shorter / Up / Down / Silence; click a pause → pickup note; play a line as synth or from the original recording. Lines can be edited, inserted and deleted in place; edits flow out of `abc` and `lyrics` |
| Score Editor | The pause between transcribing and rendering: the box fills with the incoming score on first run, you edit it on the canvas and queue again (upstream is cached). Saves the score to `output/yue2-same-music-new-lyrics/<name>.abc`. `RESET` reloads |
| Transpose | Two range levers: key_shift (whole song, semitones) and vocal_octave (sung line only, octaves); reports the vocal range before/after |
| Describe Track → Style Prompt | Audio Flamingo 3 + CLAP listen to the recording and draft a YuE2 style line (optional, ~17 GB VRAM) |
| Lyrics Fit Check | Syllables per lyric section vs sung notes per phrase; language-aware counting (English, Latin-script, Cyrillic, CJK exact-ish; Hebrew/Arabic rough; or off) |
| Set Song Language | Stamps the language as the first style tag; warns outside YuE2-3B's documented Chinese/English |
| Score Facts | bpm, key, meter, sections, nominal length from an ABC score |
| Strip Chords from Score | Chord-free score for mode `melody` |
| Compare Scores | Exact symbolic melody/rhythm check between two scores |
| Load Audio (file chooser), Load Text File | Dropdown of audio files in ComfyUI/input, ~/Downloads, ~/Music (or any path); text loader |

The heavy nodes run `workers/*.py` in the environments that `setup.sh` creates under `runtime/`
(`runtime/.venv-sheetsage2`, `runtime/.venv-describe`, weights in `runtime/models/`). Set `YUE2_RUNTIME` to relocate.

Models for the native nodes (from https://huggingface.co/Comfy-Org/YuE2):
`models/checkpoints/yue2_3b_bf16.safetensors` (or the int8 variant) and `models/audio_encoders/sheetsage2_bf16.safetensors`.

Example workflows are in `example_workflows/` (ComfyUI lists them under Templates → custom nodes).

* `stage1_transcribe_and_edit.json` — Stage 1: recording → score, edit on the canvas, saved to `output/yue2-same-music-new-lyrics/edited_score.abc`
* `stage2_generate_from_score.json` — Stage 2: load that file (or any `.abc`) → lyrics/language/style → render
* `same_music_new_lyrics.json` — both stages on one canvas with the Score Editor in the middle

`lib/abc_tools.py` is the Apache-2.0 helper from the YuE repository (see `lib/LICENSE.abc_tools`).

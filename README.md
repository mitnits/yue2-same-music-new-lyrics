# yue2-same-music-new-lyrics

Take a song, keep its melody, sing new words. Built on [YuE2](https://github.com/multimodal-art-projection/YuE):
a recording is transcribed to an editable ABC score with [SheetSage2](https://huggingface.co/m-a-p/SheetSage2),
then YuE2 renders that exact score with your lyrics, in your language, optionally an octave down or in another key.

It is a set of **ComfyUI custom nodes** (`comfyui/comfyui-yue2-same-music-new-lyrics`) around ComfyUI's built-in
YuE2 nodes: transcribe, edit the score on the canvas, fit-check the lyrics, transpose, render. Example workflows included.

Optional extras: a style-prompt drafter that listens to the recording (Audio Flamingo 3 + CLAP) and a language-aware
syllable fit check.

## Install

Linux, NVIDIA GPU (24 GB VRAM for the full-quality preset), [ComfyUI](https://github.com/comfyanonymous/ComfyUI) 0.36+,
[uv](https://docs.astral.sh/uv/), ffmpeg.

```bash
git clone https://github.com/mitnits/yue2-same-music-new-lyrics ~/yue2-same-music-new-lyrics
cd ~/yue2-same-music-new-lyrics
./setup.sh            # SheetSage2 environment and weights (~4 GB)
./setup.sh --extras   # also Audio Flamingo 3 + CLAP (~37 GB more)
ln -s ~/yue2-same-music-new-lyrics/comfyui/comfyui-yue2-same-music-new-lyrics <ComfyUI>/custom_nodes/
```

Put `yue2_3b_bf16.safetensors` from [Comfy-Org/YuE2](https://huggingface.co/Comfy-Org/YuE2) in
`<ComfyUI>/models/checkpoints/`, restart ComfyUI, and open a workflow from `comfyui/.../workflows/`
(also listed in ComfyUI's workflow sidebar once copied to `user/default/workflows/`). Set `YUE2_HOME` if this checkout
is not at `~/yue2-same-music-new-lyrics`.

## Notes

* YuE2 follows the score note for note but performs it: expect ~0.1 s timing jitter and 1–2 % tempo drift.
* Languages: YuE2 documents Chinese and English; its demo also shows Japanese, Russian, Korean, Spanish. Others are
  best effort. The language is the first tag of the style prompt.
* Model weights are CC BY-NC 4.0 (YuE2, SheetSage2) and NVIDIA non-commercial (Audio Flamingo 3): research and
  personal use only.

## License

Apache-2.0 for this repository. `yue2_studio_lib/abc_tools.py` is from the YuE repository (Apache-2.0).

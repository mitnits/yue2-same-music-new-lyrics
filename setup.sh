#!/usr/bin/env bash
# One-shot setup: YuE2 + SheetSage2 (required), and with --extras Audio Flamingo 3 + CLAP + Demucs.
# Tested on Debian 13, RTX 5090, uv 0.11. Re-running is safe: existing environments and downloads are reused.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
EXTRAS=0; [ "${1:-}" = "--extras" ] && EXTRAS=1
command -v uv >/dev/null || { echo "uv is required: https://docs.astral.sh/uv/"; exit 1; }
command -v ffmpeg >/dev/null || { echo "ffmpeg is required"; exit 1; }
HF="uvx --from huggingface-hub[hf_transfer]==0.36.2 hf"

echo "== YuE2 (repo, environment, weights)"
[ -d YuE/.git ] || git clone --depth 1 https://github.com/multimodal-art-projection/YuE.git YuE
[ -x YuE/.venv/bin/python ] || uv venv --python 3.12 YuE/.venv
uv pip install --python YuE/.venv/bin/python ./YuE gradio "huggingface-hub[cli]==0.36.2"
[ -f YuE/models/YuE2-3B/model.safetensors ] || $HF download m-a-p/YuE2-3B --local-dir YuE/models/YuE2-3B
[ -f YuE/models/YuE2-Vae/model.safetensors ] || $HF download m-a-p/YuE2-Vae --local-dir YuE/models/YuE2-Vae

echo "== SheetSage2 (own environment: Python 3.11, torch 2.8)"
[ -f YuE/models/SheetSage2/config.json ] || $HF download m-a-p/SheetSage2 --local-dir YuE/models/SheetSage2
if [ ! -x YuE/.venv-sheetsage2/bin/python ]; then
  uv venv --python 3.11 YuE/.venv-sheetsage2
  uv pip install --python YuE/.venv-sheetsage2/bin/python torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
  grep -v -E "^torch(audio)?==" YuE/models/SheetSage2/requirements.txt > /tmp/sheetsage2-req.txt
  uv pip install --python YuE/.venv-sheetsage2/bin/python -r /tmp/sheetsage2-req.txt soundfile
fi
REV=$(python3 -c "import json;print(json.load(open('YuE/models/SheetSage2/config.json'))['base_model_revision'])")
$HF download m-a-p/MERT-v2-FullSong --revision "$REV" >/dev/null

if [ "$EXTRAS" = 1 ]; then
  echo "== Extras: Audio Flamingo 3 + CLAP (style prompt drafter), Demucs (vocal separation)"
  [ -x YuE/.venv-describe/bin/python ] || uv venv --python 3.12 YuE/.venv-describe
  uv pip install --python YuE/.venv-describe/bin/python "torch==2.10.0" "transformers>=4.57" accelerate soundfile librosa numpy demucs
  [ -f YuE/models/audio-flamingo-3-hf/config.json ] || $HF download nvidia/audio-flamingo-3-hf --local-dir YuE/models/audio-flamingo-3-hf
  [ -f YuE/models/clap-music/config.json ] || $HF download laion/larger_clap_music_and_speech --local-dir YuE/models/clap-music
fi

echo
echo "Done. Start the UI with ./start_ui.sh  (add --host 0.0.0.0 to reach it from other devices)."
echo "ComfyUI: symlink comfyui/comfyui-yue2-same-music-new-lyrics into ComfyUI/custom_nodes and download"
echo "  https://huggingface.co/Comfy-Org/YuE2/resolve/main/checkpoints/yue2_3b_bf16.safetensors -> models/checkpoints/"

#!/usr/bin/env bash
# Launch the YuE2 Gradio UI. Extra args are passed through, e.g.:
#   ./start_ui.sh --host 0.0.0.0 --port 7860        (memory preset is chosen automatically)
#   ./start_ui.sh --budget 24 --quantization none     (force full BF16 preset)
cd "$(dirname "$0")"
exec YuE/.venv/bin/python yue2_ui.py "$@"

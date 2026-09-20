"""Shim: the shared helpers live in the repo root (default ~/yue2-same-music-new-lyrics, override with YUE2_HOME)."""
import os
import sys
from pathlib import Path

_home = Path(os.environ.get("YUE2_HOME", Path.home() / "yue2-same-music-new-lyrics")).expanduser()
_here = Path(__file__).resolve().parents[2]  # <repo>/comfyui/<pack> -> <repo>
for candidate in (_here, _home):
    if (candidate / "yue2_studio_lib" / "__init__.py").is_file():
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
        break
from yue2_studio_lib import abc_tools, fit, voices, transpose  # noqa: E402,F401

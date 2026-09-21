"""comfyui-yue2-same-music-new-lyrics: transcription, track description, lyrics-fit and ABC utilities around ComfyUI's YuE2 nodes."""
from .nodes import comfy_entrypoint  # noqa: F401  (ComfyUI v3 extension entry point)
from . import aligner_server

aligner_server.register_routes()

WEB_DIRECTORY = "./web"

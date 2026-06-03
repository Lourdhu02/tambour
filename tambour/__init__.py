"""Tambour - OCR engine for analog mechanical rolling-counter meters.

The reading sits on rotating digit drums (a *tambour*). This engine recognizes
the digit string from a pre-cropped strip with an SVTRv2-style CTC recognizer,
built for production scale (millions of images, multiple meter domains).

Top-level names (``build_model``, ``load_config``, ``CHARSET``, ``NUM_CLASSES``)
are resolved lazily so importing the package is cheap and never drags in torch.
"""
from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["CHARSET", "NUM_CLASSES", "load_config", "build_model", "__version__"]


def __getattr__(name):
    if name == "build_model":
        from .models import build_model
        return build_model
    if name in ("CHARSET", "NUM_CLASSES", "load_config"):
        from . import config
        return getattr(config, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

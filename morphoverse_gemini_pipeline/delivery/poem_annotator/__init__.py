"""Gemini-only gold annotation pipeline for multilingual Indian poetry."""

# Runner-repo addition (not present in the source repo): models.py does a
# top-level `from api import LLMProxyClient, LLMProxyError` against the
# sibling `morphoverse_gemini_pipeline/delivery/api.py` module. The source
# repo makes that resolve via pytest.ini's `pythonpath = .../delivery`; this
# repo also supports `python -m morphoverse_gemini_pipeline.delivery.poem_annotator....`
# invocation from the repo root, which does not put `delivery/` on sys.path
# by itself. Insert it here, once, at package-import time, so both entry
# points resolve identically without editing models.py or api.py.
import sys as _sys
from pathlib import Path as _Path

_delivery_dir = str(_Path(__file__).resolve().parents[1])
if _delivery_dir not in _sys.path:
    _sys.path.insert(0, _delivery_dir)

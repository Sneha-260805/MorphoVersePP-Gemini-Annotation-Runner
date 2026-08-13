#!/usr/bin/env python3
"""Report whether Google Cloud auth/config is resolvable — never prints a
credential, token, or key. Makes no annotation-generating call; the only
network-shaped thing it may do is ask google-auth to resolve local ADC
metadata, which google-auth itself performs without this script ever
touching the credential value.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from morphoverse_gemini_pipeline.delivery.poem_annotator import gemini_backfill_executor_v1_1 as gx  # noqa: E402


def main() -> int:
    print("Google Cloud environment check")
    print("-" * 40)

    try:
        config = gx.load_gemini_config()
        print(f"Project:  {config.project}")
        print(f"Location: {config.location}")
        print(f"Model:    {config.model}")
    except gx.ConfigError as exc:
        print(f"Configuration incomplete: {exc}")

    available = gx.check_adc_available()
    print(f"Google authentication available: {'YES' if available else 'NO'}")
    if not available:
        print("Run: gcloud auth application-default login")

    print("-" * 40)
    print("No token, key, or credential value is ever printed by this script.")
    return 0 if available else 1


if __name__ == "__main__":
    raise SystemExit(main())

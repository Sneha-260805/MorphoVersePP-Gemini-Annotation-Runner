"""Requirement coverage: (2) all required runtime imports resolve,
(1) schema loads, (3) shared prompt loads, (4) language-profile loader
works, (26) corpus runner functions without Claude/OpenAI dependencies.

No network call anywhere in this file.
"""
from __future__ import annotations

import ast
from pathlib import Path

from tests.conftest import REPO_ROOT, PROFILE_DIR

PACKAGE_DIR = REPO_ROOT / "morphoverse_gemini_pipeline" / "delivery" / "poem_annotator"
API_PY = REPO_ROOT / "morphoverse_gemini_pipeline" / "delivery" / "api.py"


def test_schema_loads():
    from morphoverse_gemini_pipeline.delivery.poem_annotator import schema
    assert schema.MORPHOVERSE_SCHEMA_VERSION == "1.1"
    assert schema.LEGACY_SCHEMA_VERSION == 5


def test_shared_prompt_loads():
    from morphoverse_gemini_pipeline.delivery.poem_annotator import shared_full_schema_prompt_v1_1 as sfp
    assert sfp.SHARED_PROMPT_CONTRACT_VERSION
    assert sfp.COMPLETENESS_CONTRACT_VERSION


def test_language_profile_loader_works():
    from morphoverse_gemini_pipeline.delivery.poem_annotator.annotation_language_profile_v1_1 import (
        load_annotation_language_profiles, get_annotation_profile_for_language,
    )
    profiles = load_annotation_language_profiles(PROFILE_DIR)
    profile = get_annotation_profile_for_language(profiles, "Hindi")
    assert profile is not None
    assert profile.profile_version == "5K.1.0"
    assert profile.native_review_required is True


def test_all_runtime_modules_import_cleanly():
    modules = [
        "alignment", "annotation_language_profile_v1_1", "backfill_plan_v1_1",
        "completeness_validator_v1_1", "config", "dataset", "execution_batch_v1_1",
        "execution_exception_v1_1", "execution_split_v1_1", "gemini_backfill_executor_v1_1",
        "gemini_pilot_execution_v1_1", "grounding", "indic_bert_context", "models", "patch_v1_1",
        "prompt_assembler_v1_1", "prompt_v1_1", "schema", "shared_full_schema_prompt_v1_1",
        "targeted_repair_prompt_v1_1", "vertex_canary_execution_v1_1", "vertex_response_schema_v1_1",
        "corpus_gemini_runner_v1_1",
    ]
    import importlib
    for name in modules:
        importlib.import_module(f"morphoverse_gemini_pipeline.delivery.poem_annotator.{name}")


def _all_py_files() -> "list[Path]":
    files = list(PACKAGE_DIR.glob("**/*.py"))
    files.append(API_PY)
    return files


def test_no_anthropic_or_openai_import_anywhere_in_package():
    forbidden = {"anthropic", "openai"}
    for f in _all_py_files():
        tree = ast.parse(f.read_text(encoding="utf-8"), filename=str(f))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top = alias.name.split(".")[0]
                    assert top not in forbidden, f"{f}: forbidden import {alias.name!r}"
            elif isinstance(node, ast.ImportFrom):
                top = (node.module or "").split(".")[0]
                assert top not in forbidden, f"{f}: forbidden import from {node.module!r}"


def test_requirements_txt_does_not_list_anthropic_or_openai():
    text = (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8").lower()
    assert "anthropic" not in text
    assert "openai" not in text


def test_no_google_genai_import_outside_gemini_backfill_executor():
    """Every provider call goes through gemini_backfill_executor_v1_1's
    client factory/retry wrapper — no second SDK import site."""
    allowed = {"gemini_backfill_executor_v1_1.py"}
    for f in PACKAGE_DIR.glob("*.py"):
        if f.name in allowed:
            continue
        tree = ast.parse(f.read_text(encoding="utf-8"), filename=str(f))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and "google" in node.module:
                raise AssertionError(f"{f}: unexpected direct google.* import outside gemini_backfill_executor_v1_1.py")

"""Shared fixtures for the runner-repo offline test suite.

No test anywhere in this directory makes a network call. Every provider
interaction goes through a fake client (dependency injection), matching the
pattern already established in the source repository's own test suite
(morphoverse_gemini_pipeline/delivery/poem_annotator/tests/test_vertex_canary_execution_v1_1.py).
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PROFILE_DIR = REPO_ROOT / "morphoverse_gemini_pipeline" / "delivery" / "poem_annotator" / "annotation_language_profiles"
SOURCE_CORPUS_DIR = REPO_ROOT / "data" / "source_corpus"


def _stanza_count_from(contents: str) -> int:
    m = re.search(r"STANZA_COUNT:\s*(\d+)", contents)
    return int(m.group(1)) if m else 1


def _detect_section(contents: str) -> str:
    m = re.search(r"SECTION TASK: (\w+)", contents)
    return m.group(1) if m else ""


def _fake_response(payload: dict):
    return SimpleNamespace(
        text=json.dumps(payload, ensure_ascii=False),
        candidates=[SimpleNamespace(finish_reason=SimpleNamespace(name="STOP"), finish_message=None)],
        usage_metadata=SimpleNamespace(prompt_token_count=10, candidates_token_count=5, total_token_count=15, cached_content_token_count=None),
        response_id="fake-response-id",
    )


def _clean_section_payload(section: str, n: int) -> dict:
    from morphoverse_gemini_pipeline.delivery.poem_annotator.prompt_assembler_v1_1 import (
        SECTION_POEM_AND_STANZA_OVERVIEW, SECTION_CULTURAL_ENTITIES,
        SECTION_FIGURATIVE_EXPRESSIONS, SECTION_TRANSLATION_LOSS, SECTION_CONSISTENCY_REVIEW,
    )
    if section == SECTION_POEM_AND_STANZA_OVERVIEW:
        return {
            "recitation_style": "lament", "emotional_arc": "grief", "theme": None,
            "stanzas": [{"index": i, "emotion": "grief", "tone": "lament", "translation_quality": "faithful", "loss_note": ""} for i in range(1, n + 1)],
            "unresolved_items": [],
        }
    if section == SECTION_CULTURAL_ENTITIES:
        return {"cultural_entities": [], "unresolved_items": []}
    if section == SECTION_FIGURATIVE_EXPRESSIONS:
        return {"stanzas": [{"index": i, "metaphor_spans": []} for i in range(1, n + 1)], "unresolved_items": []}
    if section == SECTION_TRANSLATION_LOSS:
        return {"stanzas": [{"index": i, "translation_loss": []} for i in range(1, n + 1)], "unresolved_items": []}
    if section == SECTION_CONSISTENCY_REVIEW:
        return {"consistency_findings": [], "unresolved_items": []}
    return {"unresolved_items": []}


class CleanFakeClient:
    """Always returns a schema-clean, complete response for every section.
    Records every request it receives (`.calls`) so tests can assert on
    what was actually sent, without ever touching a network socket."""

    class _Models:
        def __init__(self, outer):
            self._outer = outer

        def generate_content(self, **request):
            self._outer.calls.append(request)
            contents = request["contents"]
            section = _detect_section(contents)
            n = _stanza_count_from(contents)
            return _fake_response(_clean_section_payload(section, n))

    def __init__(self):
        self.calls: "list[dict]" = []
        self.models = self._Models(self)


class BrokenFakeClient:
    """Always returns a structurally wrong payload (missing required keys)."""

    class _Models:
        def generate_content(self, **request):
            return _fake_response({"wrong_key": "wrong_value"})

    def __init__(self):
        self.models = self._Models()


class ExplodingClientFactory:
    """A client_factory that fails the test if it is ever called — used to
    assert that --dry-run makes zero provider calls (never even constructs
    a client)."""

    def __call__(self):
        raise AssertionError("client_factory() must never be called on the dry-run path.")


@pytest.fixture
def clean_client_factory():
    client = CleanFakeClient()
    return (lambda: client), client


@pytest.fixture
def broken_client_factory():
    client = BrokenFakeClient()
    return lambda: client


@pytest.fixture
def gemini_env(monkeypatch):
    """Sets the three non-secret Gemini settings gx.load_gemini_config()
    reads (config.py evaluates them from os.environ at import time, so a
    plain monkeypatch.setenv after import wouldn't take effect — patch the
    already-imported config module's attributes directly instead)."""
    from morphoverse_gemini_pipeline.delivery.poem_annotator import config as cfg
    monkeypatch.setattr(cfg, "GOOGLE_CLOUD_PROJECT", "morphoverse-pilot-test")
    monkeypatch.setattr(cfg, "GOOGLE_CLOUD_LOCATION", "global")
    monkeypatch.setattr(cfg, "VERTEX_GEMINI_MODEL", "gemini-3.5-flash")
    yield


@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture
def profile_dir() -> Path:
    return PROFILE_DIR


def any_non_pilot_supported_poem() -> "tuple[str, str]":
    """Returns (poem_id, language) for a real, non-pilot, profile-supported
    corpus poem, read from the real exported source corpus."""
    manifest = json.loads((REPO_ROOT / "corpus" / "source_manifest.json").read_text(encoding="utf-8"))
    for r in manifest["records"]:
        if r["profile_status"] == "SUPPORTED_PILOT_VALIDATED" and r["pilot_status"] != "PILOT_ALREADY_GENERATED":
            return r["poem_id"], r["language"]
    raise AssertionError("no non-pilot, profile-supported poem found in corpus/source_manifest.json")


def any_blocked_language_poem() -> "tuple[str, str]":
    manifest = json.loads((REPO_ROOT / "corpus" / "source_manifest.json").read_text(encoding="utf-8"))
    for r in manifest["records"]:
        if r["profile_status"] != "SUPPORTED_PILOT_VALIDATED":
            return r["poem_id"], r["language"]
    raise AssertionError("no blocked-language poem found in corpus/source_manifest.json")

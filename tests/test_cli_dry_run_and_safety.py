"""Requirement coverage: (19) dry run makes zero provider calls, (25)
secrets are never persisted, plus the CLI-level safety gate that refuses to
run at all without either --dry-run or --execute --acknowledge-billing.
"""
from __future__ import annotations

import json

import pytest

from morphoverse_gemini_pipeline.delivery.poem_annotator import corpus_gemini_runner_v1_1 as runner
from tests.conftest import REPO_ROOT, any_non_pilot_supported_poem, load_release_manifest


def test_dry_run_never_constructs_a_client(monkeypatch):
    poem_id, language = any_non_pilot_supported_poem()

    def exploding_factory():
        raise AssertionError("client_factory must never be called in --dry-run.")

    monkeypatch.chdir(REPO_ROOT)
    exit_code = runner.main(
        ["--dry-run", "--poem-id", poem_id, "--language", language],
        client_factory=exploding_factory,
    )
    assert exit_code == 0


def test_cli_refuses_to_run_without_dry_run_or_execute_flags(capsys):
    with pytest.raises(SystemExit) as exc_info:
        runner.main(["--poem-id", "MV++_0001", "--language", "Hindi"])
    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert "--dry-run" in captured.err
    assert "--acknowledge-billing" in captured.err


def test_cli_refuses_execute_without_billing_ack():
    with pytest.raises(SystemExit) as exc_info:
        runner.main(["--poem-id", "MV++_0001", "--language", "Hindi", "--execute"])
    assert exc_info.value.code == 2


def test_cli_rejects_uncontrolled_concurrency():
    with pytest.raises(SystemExit):
        runner.main(["--dry-run", "--poem-id", "MV++_0001", "--language", "Hindi", "--concurrency", "999"])


def test_checkpoint_and_failure_records_never_contain_a_token_shaped_value(tmp_path, monkeypatch, clean_client_factory, gemini_env):
    """Sets a fake proxy token in the environment (mirroring a teammate's
    real .env) and asserts it never appears in any JSON this run writes —
    checkpoints, failure records, or the MODEL_CANDIDATE output itself."""
    monkeypatch.setenv("LLM_PROXY_TOKEN", "sk-not-a-real-secret-1234567890abcdef")
    poem_id, language = any_non_pilot_supported_poem()
    factory, _ = clean_client_factory
    dirs = dict(
        output_root=tmp_path / "outputs", checkpoint_dir=tmp_path / "checkpoints",
        reports_dir=tmp_path / "reports", local_run_dir=tmp_path / "local_provider_runs",
    )
    from tests.conftest import PROFILE_DIR
    result = runner.execute_poem_live(
        poem_id, language, repo_root=REPO_ROOT, profile_dir=PROFILE_DIR, client_factory=factory,
        release_manifest=load_release_manifest(), **dirs,
    )
    written_text = []
    for p in tmp_path.rglob("*.json"):
        written_text.append(p.read_text(encoding="utf-8"))
    blob = "\n".join(written_text)
    assert "sk-not-a-real-secret-1234567890abcdef" not in blob


def test_env_example_contains_no_secret_shaped_value():
    text = (REPO_ROOT / ".env.example").read_text(encoding="utf-8")
    lowered = text.lower()
    for forbidden in ("api_key", "access_token", "refresh_token", "private_key", "authorization:", "service_account"):
        assert forbidden not in lowered, f"{forbidden!r} should not appear in .env.example"

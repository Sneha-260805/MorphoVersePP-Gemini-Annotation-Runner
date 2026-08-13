from __future__ import annotations

import os

# Re-export the canonical enums so legacy `from .config import ALLOWED_*` keeps working.
from .schema import (  # noqa: F401
    ALLOWED_RECITATION_STYLES,
    ALLOWED_EMOTIONS,
    ALLOWED_TONES,
    ALLOWED_TRANSLATION_QUALITIES,
    ALLOWED_ENTITY_CATEGORIES,
)

# ── Dataset & authentication ─────────────────────────────────────────────────
DATASET_PATH = "dataset.json"
API_TOKEN = os.getenv("LLM_PROXY_TOKEN", "").strip()
DEFAULT_BASE_URL = "https://llmproxy.magnocode.tech"
TOKEN_PLACEHOLDER = "<SET_YOUR_TOKEN_HERE>"

# ── Model selection ──────────────────────────────────────────────────────────
# Claude is the primary model: the proxy's Gemini endpoint has a hard completion
# cap of ~38 tokens which truncates any annotation JSON, causing all outputs to
# fail validation.  The Claude endpoint (claude-sonnet-4-6 via proxy id "claude")
# returns full completions (361+ tokens observed).
GEMINI_PRIMARY = "claude"            # legacy one-shot path model id (only used when USE_CHUNKED_GEMINI=False)
GEMINI_FALLBACK = "gemini-3-flash"   # repair / fallback; kept as-is for retries

# ── Chunked Gemini annotation (Gemini-only) ──────────────────────────────────
# The proxy's Gemini upstreams (gemini-3.1-pro-preview / gemini-3-flash-preview)
# are thinking models with a hidden ~960-token thinking cap we cannot configure,
# so a one-shot annotation prompt truncates to empty output. The chunked path
# (poem_annotator/gemini_chunked.py) splits each poem into small single-purpose
# calls (pro first, flash fallback) so each stays under the cap. This keeps the
# pipeline Gemini-only with NO Claude/OpenAI involvement.
USE_CHUNKED_GEMINI = os.getenv("USE_CHUNKED_GEMINI", "1") == "1"
# (MODEL_ORDER and VOTING_MODELS intentionally removed — no multi-model voting.)

# ── Generation defaults ──────────────────────────────────────────────────────
REQUEST_TEMPERATURE = 0.1
# R1 fix: 1000 truncated long-poem JSON. Use a generous floor and scale by stanza count.
MAX_OUTPUT_TOKENS = 4096


def max_tokens_for(stanza_count: int) -> int:
    # Indic script metaphor_spans and cultural entity terms are token-expensive.
    # Observed: 3-stanza Punjabi annotation = 1624+ tokens (finish_reason=length).
    # Raise base to 1536 and per-stanza step to 300; cap at 2048 (proxy limit).
    return max(1536, min(2048, 1536 + 300 * max(stanza_count - 1, 0)))


# ── Alignment thresholds ─────────────────────────────────────────────────────
ALIGN_OK = 0.75            # >= this and equal counts -> "aligned"
ALIGN_RISK = 0.60          # < this -> "alignment_risk"
ALIGN_PAIR_COSINE_MIN = 0.45
USE_SEMANTIC_ALIGNMENT = os.getenv("USE_SEMANTIC_ALIGNMENT", "0") == "1"  # off by default (heavy dep)

# ── Output settings ──────────────────────────────────────────────────────────
SUMMARY_FILENAME = "annotation_summary.csv"
HUMAN_REVIEW_FILENAME = "human_review_queue.csv"
SCHEMA_VERSION = 5
PROMPT_VERSION = 9  # Gemini-only chunked annotation (gemini-3.1-pro + gemini-3-flash fallback); replaces claude one-shot

# ── Non-cultural entity filters ──────────────────────────────────────────────
NON_CULTURAL_ENTITY_TERMS: dict[str, set[str]] = {
    "Telugu": {"వసంత ఋతువు"},
}

# ── Required dataset keys ────────────────────────────────────────────────────
REQUIRED_DATASET_KEYS = (
    "language", "poem_id", "poem_title", "original_poem", "translated_poem",
)

# ── Supported languages ──────────────────────────────────────────────────────
SUPPORTED_LANGUAGES = {
    "Assamese", "Bengali", "Bodo", "Dogri", "Gujarati", "Hindi", "Kannada",
    "Kashmiri", "Konkani", "Malayalam", "Manipuri", "Marathi", "Odia",
    "Punjabi", "Rajasthani", "Sanskrit", "Santhali", "Sindhi", "Tamil",
    "Telugu", "Urdu",
}

# ── Default poem limits (per language) ──────────────────────────────────────
DEFAULT_POEMS_PER_LANGUAGE: dict[str, int] = {
    "Assamese": 5,
    "Bengali": 5,
    "Hindi": 5,
    "Marathi": 5,
    "Telugu": 5,
}


# ── Secret-safe access helpers (Stage 1 hardening) ───────────────────────────
# Additive only — does not change any existing model-call behavior or remove
# any existing name. API_TOKEN/DEFAULT_BASE_URL/etc. above are untouched.

_ENV_TOKEN_VAR = "LLM_PROXY_TOKEN"


def require_api_token() -> str:
    """Return the proxy bearer token, or raise a clear, actionable error.

    Never includes the token value itself in the error message — only the name
    of the environment variable the caller needs to set. This is the same
    resolution order main.resolve_api_token() already uses; it is duplicated
    here (not imported from main) so config-level code/tests do not need to
    import the heavier main module just to check credential presence.
    """
    env_token = os.getenv(_ENV_TOKEN_VAR, "").strip()
    if env_token:
        return env_token
    if API_TOKEN and API_TOKEN != TOKEN_PLACEHOLDER:
        return API_TOKEN
    raise RuntimeError(
        f"No API token configured. Set the {_ENV_TOKEN_VAR} environment variable "
        "before running the pipeline. Do not hardcode a token in source."
    )


# ── Reserved configuration for NOT-YET-IMPLEMENTED providers ─────────────────
# Stage 1 only documents this configuration surface (decisions: GPT, Claude,
# and Vertex-backed Gemini must eventually produce independent candidates).
# No provider client exists yet; nothing in this repository reads these values.
# Non-secret settings (model names, project id, region) are safe to read eagerly:
GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT", "").strip()
GOOGLE_CLOUD_LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1").strip()
VERTEX_GEMINI_MODEL = os.getenv("VERTEX_GEMINI_MODEL", "").strip()
GPT_MODEL_NAME = os.getenv("GPT_MODEL_NAME", "").strip()
CLAUDE_MODEL_NAME = os.getenv("CLAUDE_MODEL_NAME", "").strip()

# Reserved environment variable NAMES for future provider credentials.
# Intentionally NOT read into module-level constants: a credential should only
# be resolved at the point a provider client actually needs it (follow the
# require_api_token() pattern above), never held in an unused module global.
#   GPT_API_KEY                     - future GPT provider credential
#   CLAUDE_API_KEY                  - future Claude provider credential
#   GOOGLE_APPLICATION_CREDENTIALS  - standard Vertex AI / ADC service-account path

# Explicit allowlist of settings that are safe to log, print, or serialize.
# Anything not listed here (in particular API_TOKEN and any future *_API_KEY)
# must never appear in a logged/printed/exception-message configuration summary.
_SAFE_CONFIG_KEYS = (
    "DEFAULT_BASE_URL",
    "GEMINI_PRIMARY",
    "GEMINI_FALLBACK",
    "USE_CHUNKED_GEMINI",
    "REQUEST_TEMPERATURE",
    "MAX_OUTPUT_TOKENS",
    "ALIGN_OK",
    "ALIGN_RISK",
    "ALIGN_PAIR_COSINE_MIN",
    "USE_SEMANTIC_ALIGNMENT",
    "SCHEMA_VERSION",
    "PROMPT_VERSION",
    "GOOGLE_CLOUD_PROJECT",
    "GOOGLE_CLOUD_LOCATION",
    "VERTEX_GEMINI_MODEL",
    "GPT_MODEL_NAME",
    "CLAUDE_MODEL_NAME",
)


def safe_config_summary() -> dict[str, object]:
    """Return only the non-secret settings above — safe to log or print.

    Never includes API_TOKEN, any *_API_KEY, or any other credential, because
    it is built from an explicit allowlist rather than a blocklist.
    """
    module_globals = globals()
    return {key: module_globals[key] for key in _SAFE_CONFIG_KEYS if key in module_globals}

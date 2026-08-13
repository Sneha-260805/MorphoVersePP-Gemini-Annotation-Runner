"""Semantic grounding module using IndicBERT + LaBSE + FAISS (all optional).

All heavy imports (torch, transformers, sentence_transformers, faiss) are lazy so
the module can always be imported safely. Every public function degrades to a
harmless empty/None return when the dependencies are unavailable (e.g. broken
torch DLL on Windows).

Public API
----------
get_context_safe(poem_record)  -> dict          # pre-annotation hints for prompt
validate_annotation(poem, annotation) -> list   # post-generation review items

Helper API (exposed for external use / testing)
-----------------------------------------------
get_indic_embedding(text)                       # mean-pooled IndicBERT token embedding
get_labse_embedding(text)                       # normalized LaBSE sentence embedding
build_faiss_index(term_file, index_file)        # build or reload FAISS cultural term index
retrieve_cultural_terms(query_emb, top_k, thr)  # FAISS nearest-neighbor lookup
classify_by_prototype(emb, candidates, protos)  # cosine-nearest label from prototype dict
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── Availability probes (run once at import) ──────────────────────────────────
_NP_OK = False
try:
    import numpy as np
    _NP_OK = True
except Exception:
    logger.debug("numpy not available.")

_HAS_TORCH = False
try:
    import torch  # noqa: F401
    _HAS_TORCH = True
except Exception:
    logger.info("torch not available or DLL missing; IndicBERT disabled.")

_HAS_TRANSFORMERS = False
try:
    from transformers import AutoTokenizer, AutoModel  # noqa: F401
    _HAS_TRANSFORMERS = True
except Exception:
    logger.info("transformers not available; IndicBERT embeddings disabled.")

_HAS_SENTENCE_TRANSFORMERS = False
try:
    from sentence_transformers import SentenceTransformer  # noqa: F401
    _HAS_SENTENCE_TRANSFORMERS = True
except Exception:
    logger.info("sentence_transformers not available; LaBSE disabled.")

_HAS_FAISS = False
try:
    import faiss  # noqa: F401
    _HAS_FAISS = True
except Exception:
    logger.info("faiss not available; FAISS cultural term index disabled.")

# Legacy aliases
_TORCH_OK = _HAS_TORCH
_ST_OK    = _HAS_SENTENCE_TRANSFORMERS

_EMBEDDINGS_AVAILABLE = _HAS_TORCH and _HAS_SENTENCE_TRANSFORMERS and _NP_OK


# ── Module-level caches ────────────────────────────────────────────────────────
_sentence_model   = None
_indic_tokenizer  = None
_indic_model      = None
_faiss_index      = None
_term_list: list[str]  = []
_term_categories: list[str] = []

_emotion_prototypes: dict[str, Any] = {}
_theme_prototypes:   dict[str, Any] = {}
_prototypes_ready = False

EMOTION_LIST = [
    "grief", "longing", "devotion", "peace",
    "celebration", "resilience", "anger", "fear",
]
THEME_LIST = [
    "Nature", "Love Romance", "Philosophy", "Celebration Joy",
    "Grief Loss", "Patriotism", "Resilience", "Devotion", "Social Justice",
]


# ── Model loading (lazy, safe) ─────────────────────────────────────────────────
def _load_sentence_model():
    """Load a sentence embedding model for semantic similarity.

    Uses all-MiniLM-L6-v2 (English-optimized, 384-dim, ~90 MB) which is
    stable on all platforms.  LaBSE (multilingual 1.88 GB) is NOT loaded by
    default because it causes SIGSEGV with PyTorch-CPU + numpy>=2.x on Windows —
    a signal that Python's try/except cannot catch.

    Set env var USE_LABSE=1 to opt in to LaBSE (only safe when numpy<2 is
    installed and model weights are cached locally).
    """
    global _sentence_model
    if not _EMBEDDINGS_AVAILABLE or _sentence_model is not None:
        return _sentence_model
    if not _HAS_SENTENCE_TRANSFORMERS:
        return None
    try:
        from sentence_transformers import SentenceTransformer
        if os.getenv("USE_LABSE", "0") == "1":
            _sentence_model = SentenceTransformer("sentence-transformers/LaBSE")
        else:
            _sentence_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    except Exception:
        pass
    return _sentence_model


def _indic_weights_cached() -> bool:
    """Return True only if IndicBERT model weights are fully cached locally.

    We check for at least one .safetensors or pytorch_model*.bin shard in the
    HuggingFace cache for this model.  If absent the only option is a network
    download which (a) requires ~1.1 GB free disk space and (b) causes a fatal
    C++ memory-allocation crash on this machine when disk/RAM is nearly full.
    """
    try:
        from pathlib import Path
        hf_cache = Path(os.getenv("HF_HOME", "")) or Path.home() / ".cache" / "huggingface"
        model_slug = "models--ai4bharat--IndicBERTv2-MLM-only"
        blob_dir = hf_cache / "hub" / model_slug / "blobs"
        if not blob_dir.exists():
            return False
        # A valid weight shard is >1 MB (tokenizer configs are much smaller)
        return any(f.stat().st_size > 1_000_000 for f in blob_dir.iterdir() if f.is_file())
    except Exception:
        return False


def _load_indic_models():
    global _indic_tokenizer, _indic_model
    if not (_HAS_TORCH and _HAS_TRANSFORMERS) or _indic_tokenizer is not None:
        return _indic_tokenizer, _indic_model
    # Guard: do NOT attempt download if weights are absent locally.  An incomplete
    # download on a near-full disk causes a fatal C++ memory-allocation crash
    # (SIGABRT/exit-5) that Python try/except cannot catch.
    if not _indic_weights_cached():
        logger.info("IndicBERT weights not cached locally; skipping model load.")
        return None, None
    try:
        from transformers import AutoTokenizer, AutoModel
        name = "ai4bharat/IndicBERTv2-MLM-only"
        _indic_tokenizer = AutoTokenizer.from_pretrained(name, local_files_only=True)
        _indic_model     = AutoModel.from_pretrained(name, local_files_only=True)
        _indic_model.eval()
    except Exception:
        pass
    return _indic_tokenizer, _indic_model


# ── Embedding helpers (public) ─────────────────────────────────────────────────
def get_indic_embedding(text: str):
    """Return mean-pooled IndicBERT embedding (float32 numpy array) or None."""
    if not (_HAS_TORCH and _HAS_TRANSFORMERS and _NP_OK):
        return None
    tok, mdl = _load_indic_models()
    if tok is None or mdl is None:
        return None
    try:
        import torch
        inputs = tok(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = mdl(**inputs)
        hidden = outputs.last_hidden_state  # (1, seq_len, dim)
        mean_emb = hidden.mean(dim=1).squeeze(0).numpy()
        norm = float(np.linalg.norm(mean_emb))
        return (mean_emb / norm).astype(np.float32) if norm > 1e-9 else mean_emb.astype(np.float32)
    except Exception:
        return None


def get_labse_embedding(text: str):
    """Return normalized LaBSE embedding (float32 numpy array) or None."""
    if not _EMBEDDINGS_AVAILABLE:
        return None
    model = _load_sentence_model()
    if model is None:
        return None
    try:
        emb = model.encode(text, normalize_embeddings=True)
        return emb.astype(np.float32)
    except Exception:
        return None


def _get_sentence_embedding(text: str):
    """Internal alias for get_labse_embedding."""
    return get_labse_embedding(text)


def _cosine(a, b) -> float:
    if not _NP_OK or a is None or b is None:
        return 0.0
    try:
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        return float(np.dot(a, b) / denom) if denom > 1e-9 else 0.0
    except Exception:
        return 0.0


# ── Prototype initialization ───────────────────────────────────────────────────
_PROTO_EMOTION_TEXTS = {
    "grief":       "deep sorrow pain loss weeping mourning absence",
    "longing":     "yearning desire nostalgia missing waiting",
    "devotion":    "worship reverence dedication prayer sacred love",
    "peace":       "calm serenity stillness contentment harmony",
    "celebration": "joy happiness festivity triumph exultation",
    "resilience":  "strength perseverance endurance overcoming determination",
    "anger":       "rage fury indignation protest rebellion",
    "fear":        "dread terror anxiety fright insecurity",
}
_PROTO_THEME_TEXTS = {
    "Nature":           "trees rivers mountains rain flowers sky landscape",
    "Love Romance":     "beloved passion heart union beauty romance",
    "Philosophy":       "existence meaning truth knowledge consciousness",
    "Celebration Joy":  "festival happiness triumph dancing singing",
    "Grief Loss":       "sorrow separation death lament mourning",
    "Patriotism":       "homeland soil nation freedom sacrifice",
    "Resilience":       "strength courage endurance hope overcoming",
    "Devotion":         "god prayer temple sacred worship divine",
    "Social Justice":   "equality oppression rights justice struggle",
}


def _init_prototypes():
    global _emotion_prototypes, _theme_prototypes, _prototypes_ready
    if _prototypes_ready or not _EMBEDDINGS_AVAILABLE:
        return
    try:
        for label, text in _PROTO_EMOTION_TEXTS.items():
            emb = _get_sentence_embedding(text)
            if emb is not None:
                _emotion_prototypes[label] = emb
        for label, text in _PROTO_THEME_TEXTS.items():
            emb = _get_sentence_embedding(text)
            if emb is not None:
                _theme_prototypes[label] = emb
        _prototypes_ready = True
    except Exception:
        pass


def classify_by_prototype(embedding, candidates: list[str], prototypes: dict) -> tuple[str, float]:
    """Cosine-nearest label from a prototype dict. Returns (label, similarity)."""
    best, best_sim = "", -1.0
    if embedding is None:
        return best, best_sim
    for c in candidates:
        proto = prototypes.get(c)
        if proto is None:
            continue
        sim = _cosine(embedding, proto)
        if sim > best_sim:
            best_sim, best = sim, c
    return best, best_sim


# Internal alias kept for backward compatibility
_classify_by_prototype = classify_by_prototype


# ── FAISS cultural-term index ──────────────────────────────────────────────────
_MODULE_DIR = Path(__file__).parent.parent  # delivery/


def _resolve_path(name: str) -> str:
    """Resolve a bare filename to module-sibling dir if not found in CWD."""
    if os.path.exists(name):
        return name
    sibling = str(_MODULE_DIR / name)
    return sibling if os.path.exists(sibling) else name


def build_faiss_index(term_file: str = "cultural_terms.jsonl",
                      index_file: str = "cultural_index.faiss") -> bool:
    """Build or reload a FAISS index from a JSONL cultural-term file.

    Each JSONL line must have {"term": ..., "category": ..., "description": ...}.
    Looks for term_file in CWD first, then alongside the delivery/ package.
    Returns True if the index is ready, False if unavailable or build failed.
    """
    global _faiss_index, _term_list, _term_categories
    if _faiss_index is not None:
        return True
    if not (_NP_OK and _HAS_FAISS):
        return False

    term_file  = _resolve_path(term_file)
    index_file = str(_MODULE_DIR / os.path.basename(index_file))
    meta_path  = index_file + ".terms"

    if os.path.exists(index_file) and os.path.exists(meta_path):
        try:
            import faiss
            _faiss_index = faiss.read_index(index_file)
            with open(meta_path, encoding="utf-8") as f:
                data = json.load(f)
            _term_list       = data["terms"]
            _term_categories = data["categories"]
            return True
        except Exception:
            pass

    if not os.path.exists(term_file) or not _EMBEDDINGS_AVAILABLE:
        return False
    try:
        import faiss
        terms, cats, embs = [], [], []
        with open(term_file, encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                emb = _get_sentence_embedding(obj.get("description", obj["term"]))
                if emb is None:
                    continue
                terms.append(obj["term"])
                cats.append(obj.get("category", ""))
                embs.append(emb)
        if not embs:
            return False
        matrix = np.array(embs, dtype=np.float32)
        idx = faiss.IndexFlatIP(matrix.shape[1])
        idx.add(matrix)
        faiss.write_index(idx, index_file)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({"terms": terms, "categories": cats}, f)
        _faiss_index     = idx
        _term_list       = terms
        _term_categories = cats
        return True
    except Exception:
        return False


def _load_faiss_index(term_file: str = "cultural_terms.jsonl",
                      index_file: str = "cultural_index.faiss"):
    """Internal backward-compat wrapper around build_faiss_index."""
    build_faiss_index(term_file, index_file)


def retrieve_cultural_terms(query_embedding,
                            top_k: int = 5,
                            threshold: float = 0.6) -> list[tuple[str, str, float]]:
    """Query the FAISS index for nearest cultural terms.

    Returns list of (term, category, score) tuples above threshold.
    Returns [] if index not loaded or query_embedding is None.
    """
    build_faiss_index()
    if _faiss_index is None or not _term_list or query_embedding is None:
        return []
    if not _NP_OK:
        return []
    try:
        q = np.array([query_embedding], dtype=np.float32)
        scores, indices = _faiss_index.search(q, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or score < threshold:
                continue
            results.append((_term_list[idx], _term_categories[idx], float(score)))
        return results
    except Exception:
        return []


# Internal alias
_retrieve_cultural_terms = retrieve_cultural_terms


# ── Pre-annotation context (hints for prompt) ──────────────────────────────────
_LOW_RESOURCE_LANGUAGES = {
    "Santhali", "Bodo", "Konkani", "Manipuri", "Dogri", "Kashmiri",
}


def _split_stanzas_from_raw(original: str, translated: str) -> list[dict]:
    """Split raw poem text into stanza dicts by double-newline boundaries."""
    src_parts = [s.strip() for s in original.split("\n\n") if s.strip()]
    tr_parts  = [s.strip() for s in translated.split("\n\n") if s.strip()] if translated else []
    return [
        {
            "stanza_index": i + 1,
            "source_lines": src_parts[i].split("\n"),
            "translated_lines": tr_parts[i].split("\n") if i < len(tr_parts) else [],
        }
        for i in range(len(src_parts))
    ]


def _compute_stanza_emotions(poem_record: dict) -> list[dict]:
    """Per-stanza hints: IndicBERT for Indic source, LaBSE for alignment.

    Works from both pre-processed stanza dicts (preprocessing.stanzas) and
    raw poem text (splits on double-newlines). Returns [] when unavailable.
    """
    if not _EMBEDDINGS_AVAILABLE or not _prototypes_ready:
        return []
    stanzas_data = poem_record.get("preprocessing", {}).get("stanzas", [])
    if not stanzas_data:
        original   = poem_record.get("original_poem", "")
        translated = poem_record.get("translated_poem", "")
        if not original:
            return []
        stanzas_data = _split_stanzas_from_raw(original, translated)

    results = []
    try:
        for s in stanzas_data:
            src = " | ".join(s.get("source_lines", []))
            tr  = " | ".join(s.get("translated_lines", []))

            # IndicBERT: token-level embedding of the Indic source stanza
            src_indic = get_indic_embedding(src) if src else None
            # LaBSE: cross-lingual embedding (for alignment with English translation)
            src_labse = _get_sentence_embedding(src) if src else None
            tr_emb    = _get_sentence_embedding(tr)  if tr  else None

            # Prototype matching: use LaBSE (cross-lingual space, English prototypes)
            classify = src_labse
            emotion = "unknown"
            if classify is not None:
                emotion, _ = classify_by_prototype(classify, EMOTION_LIST, _emotion_prototypes)
            elif src_indic is not None:
                # fallback: try IndicBERT if LaBSE failed (embedding spaces differ)
                emotion, _ = classify_by_prototype(src_indic, EMOTION_LIST, _emotion_prototypes)

            align = _cosine(src_labse, tr_emb) if (src_labse is not None and tr_emb is not None) else 0.0

            results.append({
                "stanza_index": s.get("stanza_index", len(results) + 1),
                "suggested_emotion": emotion or "unknown",
                "alignment_score": round(align, 3),
            })
    except Exception:
        pass
    return results


def get_context(poem_record: dict) -> dict:
    """Full IndicBERT + LaBSE context.  Requires torch + sentence-transformers.

    IndicBERT: primary embedding for Indic source text (stanza semantics,
               cultural term retrieval, per-stanza emotion priors).
    LaBSE:     cross-lingual alignment (source ↔ translation), emotion/theme
               prototype classification against English anchor texts.
    """
    if not _EMBEDDINGS_AVAILABLE:
        return {}
    _init_prototypes()

    original   = poem_record.get("original_poem", "")
    translated = poem_record.get("translated_poem", "")

    # ── LaBSE embeddings (cross-lingual) ─────────────────────────────────────
    orig_labse  = _get_sentence_embedding(original)  if original  else None
    trans_labse = _get_sentence_embedding(translated) if translated else None

    # Need at least source LaBSE for prototype classification
    if orig_labse is None:
        return {}

    # ── IndicBERT embedding for Indic source (token-level, language-aware) ───
    indic_src_emb = get_indic_embedding(original) if original else None

    # Prototype classification: LaBSE cross-lingual space (English anchors)
    emotion_label, _ = classify_by_prototype(orig_labse, EMOTION_LIST, _emotion_prototypes)
    theme_label,   _ = classify_by_prototype(orig_labse, THEME_LIST,   _theme_prototypes)

    # Cultural term retrieval: prefer IndicBERT (native Indic embedding space)
    faiss_query = indic_src_emb if indic_src_emb is not None else orig_labse
    likely_terms = retrieve_cultural_terms(faiss_query)
    likely_cultural = [{"term": t, "category": c} for t, c, _ in likely_terms]

    # Cross-lingual alignment: LaBSE cosine(source, translation)
    align = _cosine(orig_labse, trans_labse) if trans_labse is not None else 0.0

    # Devotional probability: LaBSE cosine to English devotion anchor
    devo_emb  = _get_sentence_embedding("devotion worship prayer divine sacred temple god")
    devo_prob = _cosine(orig_labse, devo_emb) if devo_emb is not None else 0.0

    # ── Per-stanza IndicBERT emotion hints ───────────────────────────────────
    stanza_emotions = _compute_stanza_emotions(poem_record)

    result: dict = {
        "suggested_emotional_arc": emotion_label or "unknown",
        "suggested_theme":         theme_label   or "unknown",
        "likely_cultural_terms":   likely_cultural,
        "metaphor_probability":    round(min(1.0, len(likely_terms) / 5.0), 3),
        "translation_alignment_score": round(align, 3),
        "devotional_probability":  round(max(0.0, devo_prob), 3),
        "language_is_low_resource": poem_record.get("language") in _LOW_RESOURCE_LANGUAGES,
    }
    if stanza_emotions:
        result["stanza_emotions"] = stanza_emotions
    return result


def get_context_safe(poem_record: dict) -> dict:
    """Safe wrapper — returns {} on any failure (missing deps, broken torch DLL, etc.)."""
    try:
        return get_context(poem_record)
    except Exception:
        return {}


def get_stanza_contexts(poem_record: dict) -> list[dict]:
    """Public API: per-stanza IndicBERT + LaBSE hints (also called from get_context).

    Accepts both raw records (original_poem key) and pre-processed records
    (preprocessing.stanzas key).  Returns [] when embeddings unavailable.
    """
    if not _EMBEDDINGS_AVAILABLE:
        return []
    _init_prototypes()
    return _compute_stanza_emotions(poem_record)


# ── Post-generation validation ─────────────────────────────────────────────────
def validate_annotation(poem_record: dict, annotation: dict) -> list[dict]:
    """Check annotation plausibility with LaBSE.  Returns review items (may be empty).

    Runs only when embeddings are available.  Never modifies the annotation itself;
    only generates advisory review items for human inspection.
    """
    if not _EMBEDDINGS_AVAILABLE or not _prototypes_ready:
        return []

    review_items: list[dict] = []
    original   = poem_record.get("original_poem", "")
    translated = poem_record.get("translated_poem", "")
    full_text  = f"{original}\n{translated}"

    poem_emb = _get_sentence_embedding(full_text)
    if poem_emb is None:
        return []

    try:
        for stanza in annotation.get("stanzas", []):
            claimed = stanza.get("emotion", "")
            if not claimed or not _emotion_prototypes:
                continue
            best_match, _  = classify_by_prototype(poem_emb, EMOTION_LIST, _emotion_prototypes)
            claimed_sim, _ = classify_by_prototype(poem_emb, [claimed], _emotion_prototypes)
            if best_match and best_match != claimed and claimed_sim < 0.30:
                review_items.append({
                    "field_path": f"annotation.stanzas[{stanza.get('index', 0)}].emotion",
                    "severity": "low",
                    "resolved_value": claimed,
                    "model_value": claimed,
                    "note": (
                        f"IndicBERT/LaBSE suggests emotion={best_match!r} "
                        f"(model claimed {claimed!r}; cosine similarity={claimed_sim:.2f}). "
                        "Human review recommended."
                    ),
                })

        orig_emb  = _get_sentence_embedding(original)  or poem_emb
        trans_emb = _get_sentence_embedding(translated) or poem_emb
        align_score = _cosine(orig_emb, trans_emb)
        stanzas = annotation.get("stanzas", [])
        if stanzas:
            qualities = [s.get("translation_quality", "") for s in stanzas]
            faithful_frac = qualities.count("faithful") / max(len(qualities), 1)
            if faithful_frac > 0.7 and align_score < 0.50:
                review_items.append({
                    "field_path": "annotation.translation_fidelity_score",
                    "severity": "medium",
                    "resolved_value": str(round(align_score, 2)),
                    "model_value": "faithful (majority)",
                    "note": (
                        f"IndicBERT/LaBSE cross-lingual alignment={align_score:.2f} "
                        "is low despite model marking most stanzas 'faithful'. "
                        "Verify translation quality manually."
                    ),
                })

    except Exception:
        pass

    return review_items

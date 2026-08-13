"""Stanza alignment confidence scoring.

Blank-line splitting (in dataset.py) is the segmentation primitive. This module
*scores how trustworthy that segmentation is* rather than replacing it, so the
pipeline can refuse stanza-level judgments when alignment is risky.
"""
from __future__ import annotations

from dataclasses import dataclass

from .config import ALIGN_OK, ALIGN_RISK, ALIGN_PAIR_COSINE_MIN, USE_SEMANTIC_ALIGNMENT
from .schema import ALIGNMENT_OK, ALIGNMENT_LOW, ALIGNMENT_RISK


@dataclass(frozen=True)
class AlignmentResult:
    status: str            # aligned | aligned_low | alignment_risk
    confidence: float      # 0.0 .. 1.0
    note: str
    min_pair_cosine: float


def _char_len(lines: list[str]) -> int:
    return sum(len(x) for x in lines)


def _length_ratio(a: list[str], b: list[str]) -> float:
    la, lb = _char_len(a), _char_len(b)
    if la == 0 and lb == 0:
        return 1.0
    if la == 0 or lb == 0:
        return 0.0
    return min(la, lb) / max(la, lb)


def _semantic_pair_cosines(source_stanzas, translated_stanzas) -> list[float]:
    """Optional LaBSE cosine per aligned pair. Returns [] if embeddings unavailable."""
    if not USE_SEMANTIC_ALIGNMENT:
        return []
    try:
        from .indic_bert_context import get_sentence_embedding
        import numpy as np
    except Exception:
        return []
    cosines: list[float] = []
    for s, t in zip(source_stanzas, translated_stanzas):
        s_text, t_text = " ".join(s), " ".join(t)
        if not s_text.strip() or not t_text.strip():
            cosines.append(0.0)
            continue
        try:
            es = get_sentence_embedding(s_text)
            et = get_sentence_embedding(t_text)
            denom = (np.linalg.norm(es) * np.linalg.norm(et)) or 1e-9
            cosines.append(float(np.dot(es, et) / denom))
        except Exception:
            return []  # degrade gracefully — fall back to length-only scoring
    return cosines


def align_stanzas(source_stanzas: list[list[str]], translated_stanzas: list[list[str]]) -> AlignmentResult:
    src_n, tr_n = len(source_stanzas), len(translated_stanzas)

    # 1) count score
    if src_n == tr_n:
        count_score = 1.0
    else:
        count_score = max(0.0, 1.0 - abs(src_n - tr_n) / max(src_n, 1))

    # pair up to the shorter length for length/semantic checks
    pair_count = min(src_n, tr_n)
    src_pairs = source_stanzas[:pair_count]
    tr_pairs = translated_stanzas[:pair_count]

    # 2) length score
    if pair_count:
        length_score = sum(_length_ratio(a, b) for a, b in zip(src_pairs, tr_pairs)) / pair_count
    else:
        length_score = 0.0

    # 3) semantic score (optional)
    cosines = _semantic_pair_cosines(src_pairs, tr_pairs)
    if cosines:
        semantic_score = sum(cosines) / len(cosines)
        min_pair_cosine = min(cosines)
    else:
        semantic_score = length_score   # fall back to length when embeddings are off
        min_pair_cosine = 1.0

    confidence = round(0.4 * count_score + 0.2 * length_score + 0.4 * semantic_score, 4)

    risky = (
        confidence < ALIGN_RISK
        or abs(src_n - tr_n) > 1
        or (cosines and min_pair_cosine < ALIGN_PAIR_COSINE_MIN)
    )
    if risky:
        status = ALIGNMENT_RISK
        note = (f"Low alignment confidence ({confidence}); source={src_n} translated={tr_n}; "
                f"min_pair_cosine={round(min_pair_cosine, 3)}. Stanza-level labels not trusted.")
    elif src_n == tr_n and confidence >= ALIGN_OK:
        status = ALIGNMENT_OK
        note = "Stanza counts match and alignment confidence is high."
    else:
        status = ALIGNMENT_LOW
        note = f"Alignment acceptable but not high (confidence={confidence})."

    return AlignmentResult(status=status, confidence=confidence, note=note, min_pair_cosine=round(min_pair_cosine, 4))

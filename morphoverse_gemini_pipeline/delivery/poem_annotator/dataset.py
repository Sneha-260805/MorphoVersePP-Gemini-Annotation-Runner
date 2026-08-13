from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .config import REQUIRED_DATASET_KEYS, SUPPORTED_LANGUAGES
from .alignment import align_stanzas


class DatasetValidationError(ValueError):
    """Raised when the dataset structure is invalid."""


@dataclass(frozen=True)
class StanzaInput:
    stanza_index: int
    line_count: int
    source_lines: list[str]
    translated_lines: list[str]


@dataclass(frozen=True)
class PreprocessedPoem:
    poem_id: str
    poem_title: str
    language: str
    original_poem: str
    translated_poem: str
    stanzas: list[StanzaInput]
    source_stanza_count: int
    translated_stanza_count: int
    alignment_status: str
    alignment_confidence: float
    alignment_note: str


def resolve_dataset_path(path_like: str | Path) -> Path:
    candidate = Path(path_like)
    if candidate.exists():
        return candidate
    if candidate.name == "dataset.json":
        alt = candidate.with_name("dataset.JSON")
        if alt.exists():
            return alt
    elif candidate.name == "dataset.JSON":
        alt = candidate.with_name("dataset.json")
        if alt.exists():
            return alt
    raise FileNotFoundError(f"Dataset file not found: {candidate}")


def load_dataset_file(dataset_path: Path) -> list[dict[str, str]]:
    with dataset_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, dict) and "poems" in payload and isinstance(payload["poems"], list):
        payload = payload["poems"]
    elif not isinstance(payload, list):
        raise DatasetValidationError("Dataset JSON must contain a list of poem objects.")

    records: list[dict[str, str]] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise DatasetValidationError(f"Dataset row {index} is not an object.")
        missing = [key for key in REQUIRED_DATASET_KEYS if key not in item]
        if missing:
            raise DatasetValidationError(f"Dataset row {index} is missing keys: {missing}")
        record: dict[str, str] = {}
        for key in REQUIRED_DATASET_KEYS:
            value = item[key]
            if not isinstance(value, str):
                raise DatasetValidationError(f"Dataset row {index} field '{key}' must be a string.")
            record[key] = value
        if record["language"] not in SUPPORTED_LANGUAGES:
            raise DatasetValidationError(f"Unsupported language in dataset row {index}: {record['language']}")
        records.append(record)
    return records


def index_dataset_by_language(dataset: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    indexed: dict[str, list[dict[str, str]]] = {language: [] for language in SUPPORTED_LANGUAGES}
    for record in dataset:
        indexed.setdefault(record["language"], []).append(record)
    return indexed


def normalize_poem_text(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def drop_exact_title_line(text: str, title: str) -> str:
    normalized = normalize_poem_text(text)
    lines = normalized.split("\n")
    if not lines:
        return normalized
    first_content_index = next((i for i, line in enumerate(lines) if line.strip()), None)
    if first_content_index is None:
        return normalized
    if lines[first_content_index].strip().casefold() != title.strip().casefold():
        return normalized
    reduced = lines[:first_content_index] + lines[first_content_index + 1:]
    return "\n".join(reduced).strip()


def split_stanzas(text: str) -> list[list[str]]:
    normalized = normalize_poem_text(text)
    if not normalized:
        return []
    stanzas: list[list[str]] = []
    current: list[str] = []
    for raw_line in normalized.split("\n"):
        line = raw_line.strip()
        if not line:
            if current:
                stanzas.append(current)
                current = []
            continue
        current.append(line)
    if current:
        stanzas.append(current)
    return stanzas


def preprocess_poem(record: dict[str, str]) -> PreprocessedPoem:
    original_text = drop_exact_title_line(record["original_poem"], record["poem_title"])
    translated_text = drop_exact_title_line(record["translated_poem"], record["poem_title"])

    source_stanzas = split_stanzas(original_text)
    translated_stanzas = split_stanzas(translated_text)
    if not source_stanzas:
        raise DatasetValidationError(f"Poem {record['poem_id']} has no source stanzas after preprocessing.")

    translated_stanza_count = len(translated_stanzas)

    # Score alignment BEFORE we pad/truncate, so confidence reflects the real mismatch.
    align = align_stanzas(source_stanzas, translated_stanzas)

    # Pad/truncate the translation to match source count for 1:1 stanza objects.
    if len(translated_stanzas) < len(source_stanzas):
        translated_stanzas = translated_stanzas + [[] for _ in range(len(source_stanzas) - len(translated_stanzas))]
    elif len(translated_stanzas) > len(source_stanzas):
        translated_stanzas = translated_stanzas[:len(source_stanzas)]

    stanzas = [
        StanzaInput(
            stanza_index=idx,
            line_count=len(source_lines),
            source_lines=source_lines,
            translated_lines=translated_stanzas[idx - 1],
        )
        for idx, source_lines in enumerate(source_stanzas, start=1)
    ]

    return PreprocessedPoem(
        poem_id=record["poem_id"],
        poem_title=record["poem_title"],
        language=record["language"],
        original_poem=original_text,
        translated_poem=translated_text,
        stanzas=stanzas,
        source_stanza_count=len(source_stanzas),
        translated_stanza_count=translated_stanza_count,
        alignment_status=align.status,
        alignment_confidence=align.confidence,
        alignment_note=align.note,
    )

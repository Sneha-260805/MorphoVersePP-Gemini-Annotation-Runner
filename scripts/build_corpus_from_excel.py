#!/usr/bin/env python3
"""Stage 5M.1 — one-time, offline corpus-export tool.

Reads the authoritative raw workbook (data/raw/Indian_poem_dataset.xlsx —
the SHA-256-verified source of truth for the MorphoVerse++ corpus, see
DATASET_PROVENANCE.md) and (re)generates every derived, deterministic
corpus artifact this repo's runner actually reads at execution time:

  - data/source_corpus/<language>/MV++_XXXX.json   (5-field generation input)
  - corpus/corpus_metadata_manifest.json            (poet/translator/titles —
    NOT a generation input, kept structurally separate)
  - corpus/corpus_inventory.json / .csv
  - corpus/language_inventory.json
  - corpus/source_manifest.json
  - corpus/trailing_unlabeled_records.json / TRAILING_RECORD_AUDIT.md

Uses only the Python standard library (zipfile + xml.etree) to parse the
.xlsx — openpyxl/pandas are not installed in the team environment this was
built in, and installing them would require a network call, which this tool
must never make. It never calls Vertex/Gemini, Claude, or OpenAI, and it
makes no network request of any kind.

Run once, from the repo root:

    python scripts/build_corpus_from_excel.py

Re-run any time the raw workbook is intentionally updated; every output
listed above is fully regenerated (not incrementally patched) so results
never depend on run order or prior state.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_XLSX = REPO_ROOT / "data" / "raw" / "Indian_poem_dataset.xlsx"
SOURCE_CORPUS_DIR = REPO_ROOT / "data" / "source_corpus"
CORPUS_DIR = REPO_ROOT / "corpus"

NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
CELL_REF_RE = re.compile(r"^([A-Z]+)(\d+)$")

# Must equal poem_annotator/config.py's SUPPORTED_LANGUAGES exactly — this
# is also the mechanism that discovers the embedded, mid-sheet duplicate
# header row (its Language-column value, "Original Language", is not a
# member of this set) without hardcoding any row number.
SUPPORTED_LANGUAGES = {
    "Assamese", "Bengali", "Bodo", "Dogri", "Gujarati", "Hindi", "Kannada",
    "Kashmiri", "Konkani", "Malayalam", "Manipuri", "Marathi", "Odia",
    "Punjabi", "Rajasthani", "Sanskrit", "Santhali", "Sindhi", "Tamil",
    "Telugu", "Urdu",
}

PROFILE_SUPPORTED_LANGUAGES = {"Bengali", "Hindi", "Kannada", "Kashmiri", "Sindhi", "Telugu"}

PILOT_POEM_IDS = {"MV++_0011", "MV++_0073", "MV++_1118", "MV++_1153", "MV++_1249", "MV++_1443"}

EXPECTED_HEADER = ["Language", "Poem Title", "Translated Title", "Poet", "Translator", "Original Poems", "Translated Poems"]


# ══════════════════════════════════════════════════════════════════════════
# Minimal stdlib-only .xlsx reader (see docstring above for why)
# ══════════════════════════════════════════════════════════════════════════
def _col_to_index(col_letters: str) -> int:
    idx = 0
    for ch in col_letters:
        idx = idx * 26 + (ord(ch) - ord("A") + 1)
    return idx - 1


def _load_shared_strings(z: zipfile.ZipFile) -> "list[str]":
    if "xl/sharedStrings.xml" not in z.namelist():
        return []
    root = ET.fromstring(z.read("xl/sharedStrings.xml"))
    strings = []
    for si in root.findall("m:si", NS):
        texts = [t.text or "" for t in si.iter("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t")]
        strings.append("".join(texts))
    return strings


def read_first_sheet_rows(xlsx_path: Path) -> "list[list[str]]":
    z = zipfile.ZipFile(xlsx_path)
    sheet_files = sorted(n for n in z.namelist() if n.startswith("xl/worksheets/sheet") and n.endswith(".xml"))
    if not sheet_files:
        raise RuntimeError(f"{xlsx_path}: no worksheet found in the workbook.")
    sheet_file = sheet_files[0]

    shared = _load_shared_strings(z)
    root = ET.fromstring(z.read(sheet_file))
    sheet_data = root.find("m:sheetData", NS)

    row_records = []
    max_col = 0
    for row_el in sheet_data.findall("m:row", NS):
        row_num = int(row_el.get("r"))
        row_cells: "dict[int, str]" = {}
        for c in row_el.findall("m:c", NS):
            ref = c.get("r")
            m = CELL_REF_RE.match(ref)
            col_idx = _col_to_index(m.group(1))
            cell_type = c.get("t")
            value = ""
            if cell_type == "s":
                v_el = c.find("m:v", NS)
                if v_el is not None and v_el.text is not None:
                    value = shared[int(v_el.text)]
            elif cell_type == "inlineStr":
                is_el = c.find("m:is", NS)
                if is_el is not None:
                    value = "".join(t.text or "" for t in is_el.iter("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t"))
            else:
                v_el = c.find("m:v", NS)
                if v_el is not None and v_el.text is not None:
                    value = v_el.text
            row_cells[col_idx] = value
            max_col = max(max_col, col_idx)
        row_records.append((row_num, row_cells))

    if not row_records:
        return []
    max_row = max(r for r, _ in row_records)
    grid = [["" for _ in range(max_col + 1)] for _ in range(max_row)]
    for row_num, cells in row_records:
        for col_idx, value in cells.items():
            grid[row_num - 1][col_idx] = value
    return grid


def normalize_text(t: str) -> str:
    """Whitespace-only normalization (CRLF -> LF, strip outer whitespace) —
    the same normalization poem_annotator/dataset.py's own
    normalize_poem_text() already performs on every poem downstream. Never
    touches interior content, spelling, or translation choices."""
    return (t or "").replace("\r\n", "\n").strip()


# ══════════════════════════════════════════════════════════════════════════
# Canonical corpus extraction
# ══════════════════════════════════════════════════════════════════════════
def extract_canonical_corpus(rows: "list[list[str]]") -> dict:
    if not rows:
        raise RuntimeError("Workbook has no rows.")

    header = [normalize_text(c) for c in rows[0][:7]]
    if header != EXPECTED_HEADER:
        raise RuntimeError(f"Unexpected header row: {header!r} (expected {EXPECTED_HEADER!r}). "
                            "Re-run the audit before trusting column positions.")

    canonical: "list[dict]" = []
    anomalous_rows: "list[dict]" = []   # embedded headers / any non-blank, non-supported language value
    blank_language_rows: "list[dict]" = []  # trailing (or any) poem-like rows with no Language value

    next_id = 1
    for excel_0idx in range(1, len(rows)):
        r = rows[excel_0idx]
        r = r + [""] * (7 - len(r)) if len(r) < 7 else r
        language = normalize_text(r[0])
        has_any_content = any(normalize_text(c) for c in r[:7])
        if not has_any_content:
            continue

        record = {
            "excel_row": excel_0idx + 1,
            "language": language,
            "poem_title": normalize_text(r[1]),
            "translated_title": normalize_text(r[2]),
            "poet": normalize_text(r[3]),
            "translator": normalize_text(r[4]),
            "original_poem": normalize_text(r[5]),
            "translated_poem": normalize_text(r[6]),
        }

        if language == "":
            blank_language_rows.append(record)
            continue
        if language not in SUPPORTED_LANGUAGES:
            record["observed_language_value"] = language
            anomalous_rows.append(record)
            continue

        poem_id = f"MV++_{next_id:04d}"
        record["poem_id"] = poem_id
        canonical.append(record)
        next_id += 1

    return {
        "canonical": canonical,
        "anomalous_rows": anomalous_rows,
        "blank_language_rows": blank_language_rows,
    }


# ══════════════════════════════════════════════════════════════════════════
# Output writers
# ══════════════════════════════════════════════════════════════════════════
def write_source_corpus(canonical: "list[dict]") -> None:
    if SOURCE_CORPUS_DIR.exists():
        # File-by-file removal (not shutil.rmtree on the whole tree at once)
        # — on this OneDrive-synced checkout, rmtree can hit a transient
        # WinError 5 on a directory OneDrive still has a handle open on.
        for f in SOURCE_CORPUS_DIR.glob("**/*"):
            if f.is_file():
                f.unlink()
        for d in sorted(SOURCE_CORPUS_DIR.glob("**/*"), key=lambda p: len(p.parts), reverse=True):
            if d.is_dir():
                d.rmdir()
    SOURCE_CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    for rec in canonical:
        out_dir = SOURCE_CORPUS_DIR / rec["language"]
        out_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "poem_id": rec["poem_id"],
            "poem_title": rec["poem_title"],
            "language": rec["language"],
            "original_poem": rec["original_poem"],
            "translated_poem": rec["translated_poem"],
        }
        (out_dir / f"{rec['poem_id']}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
        )


def write_metadata_manifest(canonical: "list[dict]") -> None:
    entries = [
        {
            "poem_id": rec["poem_id"], "language": rec["language"],
            "poem_title": rec["poem_title"], "translated_title": rec["translated_title"],
            "poet": rec["poet"], "translator": rec["translator"],
            "excel_row": rec["excel_row"],
        }
        for rec in canonical
    ]
    json.dump(
        {"note": "Poet/translator/title metadata — NOT a generation input; kept separate from data/source_corpus/.",
         "total_records": len(entries), "records": entries},
        open(CORPUS_DIR / "corpus_metadata_manifest.json", "w", encoding="utf-8"),
        indent=2, ensure_ascii=False,
    )


def write_source_manifest_and_inventory(canonical: "list[dict]") -> dict:
    def sha256_text(t: str) -> str:
        return hashlib.sha256(t.encode("utf-8")).hexdigest()

    manifest_entries = []
    per_language_counts: "dict[str, int]" = {}
    per_language_ids: "dict[str, list[str]]" = {}
    seen_ids: "dict[str, str]" = {}
    duplicate_ids: "list[str]" = []
    empty_poems: "list[str]" = []
    missing_translations: "list[str]" = []

    for rec in canonical:
        pid = rec["poem_id"]
        lang = rec["language"]
        if pid in seen_ids:
            duplicate_ids.append(pid)
        else:
            seen_ids[pid] = rec["excel_row"]
        if not rec["original_poem"]:
            empty_poems.append(pid)
        if not rec["translated_poem"]:
            missing_translations.append(pid)

        per_language_counts[lang] = per_language_counts.get(lang, 0) + 1
        per_language_ids.setdefault(lang, []).append(pid)

        manifest_entries.append({
            "poem_id": pid, "language": lang,
            "source_path": f"data/source_corpus/{lang}/{pid}.json",
            "excel_row": rec["excel_row"],
            "original_poem_sha256": sha256_text(rec["original_poem"]),
            "translation_sha256": sha256_text(rec["translated_poem"]),
            "pilot_status": "PILOT_ALREADY_GENERATED" if pid in PILOT_POEM_IDS else "AVAILABLE_FOR_ASSIGNMENT",
            "profile_status": "SUPPORTED_PILOT_VALIDATED" if lang in PROFILE_SUPPORTED_LANGUAGES else "PROFILE_MISSING",
        })

    manifest_entries.sort(key=lambda e: e["poem_id"])
    json.dump(
        {
            "generated_from": "data/raw/Indian_poem_dataset.xlsx (this repo's own copy — see DATASET_PROVENANCE.md)",
            "total_records": len(manifest_entries),
            "records": manifest_entries,
        },
        open(CORPUS_DIR / "source_manifest.json", "w", encoding="utf-8"),
        indent=2, ensure_ascii=False,
    )

    inventory = {
        "total_poem_count": len(manifest_entries),
        "total_language_count": len(per_language_counts),
        "poems_per_language": dict(sorted(per_language_counts.items())),
        "missing_translations": missing_translations,
        "duplicate_poem_ids": sorted(set(duplicate_ids)),
        "malformed_records": [],
        "empty_poems": empty_poems,
        "pilot_poem_ids": sorted(PILOT_POEM_IDS & set(seen_ids.keys())),
        "pilot_poem_ids_not_found_in_source": sorted(PILOT_POEM_IDS - set(seen_ids.keys())),
        "all_poem_ids_unique": len(duplicate_ids) == 0,
        "source_root": "data/raw/Indian_poem_dataset.xlsx (authoritative workbook — recompute, do not hardcode)",
    }
    json.dump(inventory, open(CORPUS_DIR / "corpus_inventory.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    with open(CORPUS_DIR / "corpus_inventory.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["poem_id", "language", "pilot_status", "profile_status", "source_path"])
        for e in manifest_entries:
            writer.writerow([e["poem_id"], e["language"], e["pilot_status"], e["profile_status"], e["source_path"]])

    json.dump(
        {lang: {"poem_count": count, "poem_ids": sorted(per_language_ids[lang])}
         for lang, count in sorted(per_language_counts.items())},
        open(CORPUS_DIR / "language_inventory.json", "w", encoding="utf-8"),
        indent=2, ensure_ascii=False,
    )
    return inventory


def write_trailing_record_audit(blank_language_rows: "list[dict]", anomalous_rows: "list[dict]") -> None:
    entries = []
    for rec in blank_language_rows:
        entries.append({
            "excel_row": rec["excel_row"],
            "title": rec["poem_title"],
            "translated_title": rec["translated_title"],
            "poet": rec["poet"],
            "translator": rec["translator"],
            "script_observation": "non-Latin script present" if any(ord(ch) > 127 for ch in rec["original_poem"]) else "ASCII-only / undetermined",
            "reason_excluded": "Outside canonical 1,570-record boundary and missing required Language field.",
            "classification": "TRAILING_UNLABELED_EXTRA_RECORD",
        })
    json.dump(
        {"total_trailing_records": len(entries), "records": entries},
        open(CORPUS_DIR / "trailing_unlabeled_records.json", "w", encoding="utf-8"),
        indent=2, ensure_ascii=False,
    )

    lines = [
        "# Trailing unlabeled record audit\n",
        f"{len(entries)} poem-like row(s) were found immediately after the canonical "
        "1,570-record corpus boundary, each missing a required `Language` value. They "
        "are preserved here for audit only — never assigned an MV++ ID, never included "
        "in the source-only export, and never included in any teammate assignment.\n",
        "| Excel row | Title | Translated title | Poet | Script observation | Reason excluded |",
        "|---|---|---|---|---|---|",
    ]
    for e in entries:
        lines.append(f"| {e['excel_row']} | {e['title']} | {e['translated_title']} | {e['poet']} | {e['script_observation']} | {e['reason_excluded']} |")

    if anomalous_rows:
        lines.append("\n## Embedded/anomalous rows excluded from the canonical count\n")
        lines.append("Rows whose `Language` cell is non-blank but is not one of the 21 "
                      "supported language names — this is how a mid-sheet duplicate header "
                      "row is discovered programmatically, without hardcoding a row number.\n")
        lines.append("| Excel row | Observed language-column value | Col B | Col C |")
        lines.append("|---|---|---|---|")
        for r in anomalous_rows:
            lines.append(f"| {r['excel_row']} | {r['observed_language_value']} | {r['poem_title']} | {r['translated_title']} |")

    open(CORPUS_DIR / "TRAILING_RECORD_AUDIT.md", "w", encoding="utf-8").write("\n".join(lines) + "\n")


# ══════════════════════════════════════════════════════════════════════════
def main() -> int:
    if not RAW_XLSX.exists():
        raise SystemExit(f"Raw workbook not found at {RAW_XLSX}. See DATASET_PROVENANCE.md.")

    rows = read_first_sheet_rows(RAW_XLSX)
    result = extract_canonical_corpus(rows)
    canonical = result["canonical"]

    if len(canonical) != 1570:
        raise SystemExit(
            f"STOP: independent audit found {len(canonical)} canonical poems, not the "
            "expected 1,570. Refusing to rewrite the teammate repository — investigate "
            "the workbook structure before proceeding."
        )

    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    write_source_corpus(canonical)
    write_metadata_manifest(canonical)
    inventory = write_source_manifest_and_inventory(canonical)
    write_trailing_record_audit(result["blank_language_rows"], result["anomalous_rows"])

    print(f"Canonical poems written: {len(canonical)}")
    print(f"Languages: {inventory['total_language_count']}")
    print(f"Trailing unlabeled records: {len(result['blank_language_rows'])}")
    print(f"Anomalous/embedded-header rows excluded: {len(result['anomalous_rows'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

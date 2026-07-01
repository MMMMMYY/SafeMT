import csv
import json
from pathlib import Path
from typing import Iterable

from utils_text import normalize_whitespace, resolve_path


SUPPORTED_LOCAL_SUFFIXES = {".csv", ".json", ".jsonl", ".ndjson", ".parquet", ".txt", ".tsv"}
PROMPT_COLUMNS = ("prompt", "text", "caption", "query", "instruction", "input")
CATEGORY_COLUMNS = ("category", "label", "task", "class", "safety_category", "risk_category")


def iter_data_files(path: str | Path) -> list[Path]:
    root = resolve_path(path)
    if not root.exists():
        return []
    if root.is_file():
        return [root] if root.suffix.lower() in SUPPORTED_LOCAL_SUFFIXES else []
    return sorted(
        file_path
        for file_path in root.rglob("*")
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_LOCAL_SUFFIXES
    )


def read_local_records(path: str | Path, max_records: int | None = None) -> list[dict]:
    records: list[dict] = []
    for file_path in iter_data_files(path):
        for record in read_records_from_file(file_path):
            records.append(record)
            if max_records and len(records) >= max_records:
                return records
    return records


def read_records_from_file(path: str | Path) -> list[dict]:
    file_path = resolve_path(path)
    suffix = file_path.suffix.lower()
    if suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        with file_path.open("r", encoding="utf-8", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle, delimiter=delimiter)]
    if suffix in {".jsonl", ".ndjson"}:
        records = []
        with file_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    value = json.loads(line)
                    if isinstance(value, dict):
                        records.append(value)
        return records
    if suffix == ".txt":
        with file_path.open("r", encoding="utf-8") as handle:
            return [{"text": line.strip()} for line in handle if line.strip()]
    if suffix == ".json":
        with file_path.open("r", encoding="utf-8") as handle:
            return coerce_json_records(json.load(handle))
    if suffix == ".parquet":
        try:
            import pandas as pd
        except ImportError as exc:
            raise RuntimeError(
                f"Cannot read Parquet file {file_path}. Install pandas and pyarrow, or export as CSV/JSONL."
            ) from exc
        return pd.read_parquet(file_path).to_dict(orient="records")
    return []


def coerce_json_records(value) -> list[dict]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if isinstance(value, dict):
        for key in ("records", "data", "rows", "items", "train", "validation", "test"):
            nested = value.get(key)
            if isinstance(nested, list):
                return [item for item in nested if isinstance(item, dict)]
        combined = []
        for nested in value.values():
            if isinstance(nested, list):
                combined.extend(item for item in nested if isinstance(item, dict))
        if combined:
            return combined
        return [value]
    return []


def find_likely_column(record: dict, candidates: Iterable[str]) -> str | None:
    lowered = {str(key).lower().strip(): key for key in record.keys()}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    for candidate in candidates:
        for lowered_key, original_key in lowered.items():
            if candidate in lowered_key:
                return original_key
    return None


def find_likely_text(record: dict, candidates: Iterable[str]) -> str:
    column = find_likely_column(record, candidates)
    if not column:
        return ""
    value = record.get(column, "")
    if value is None:
        return ""
    return normalize_whitespace(str(value))


def metadata_without(record: dict, columns: Iterable[str | None]) -> dict:
    excluded = {column for column in columns if column}
    metadata = {}
    for key, value in record.items():
        if key not in excluded:
            metadata[str(key)] = value
    return metadata

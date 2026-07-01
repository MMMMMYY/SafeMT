import argparse
from pathlib import Path
from typing import Iterable

from data_io import CATEGORY_COLUMNS, PROMPT_COLUMNS, find_likely_column, find_likely_text, metadata_without, read_local_records
from utils_text import load_json, resolve_path, sentence_case, strip_style_words, write_jsonl


DEFAULT_SENSITIVE_PATHS = {
    "T2ISafety": "data/raw/sensitive/t2i_safety",
    "I2P": "data/raw/sensitive/i2p",
    "T2I-RiskyPrompt": "data/raw/sensitive/t2i_risky_prompt",
}

HF_DATASETS = {
    "T2ISafety": "OpenSafetyLab/t2i_safety_dataset",
    "I2P": "AIML-TUDA/i2p",
    "T2I-RiskyPrompt": "datarr/T2I-RiskyPrompt-ImageDataset",
}


def normalize_category_text(value: str) -> str:
    return value.lower().replace("_", " ").replace("-", " ").strip()


def map_category(raw_category: str, prompt: str, category_mapping: dict[str, list[str]]) -> str:
    raw_text = normalize_category_text(raw_category)
    prompt_text = normalize_category_text(prompt)
    for category, terms in category_mapping.items():
        if raw_text and any(normalize_category_text(term) in raw_text for term in terms):
            return category
    for category, terms in category_mapping.items():
        if any(normalize_category_text(term) in prompt_text for term in terms):
            return category
    return ""


def unified_sensitive_record(raw_record: dict, source_dataset: str, category_mapping: dict[str, list[str]]) -> dict | None:
    prompt_column = find_likely_column(raw_record, PROMPT_COLUMNS)
    category_column = find_likely_column(raw_record, CATEGORY_COLUMNS)
    source_prompt = find_likely_text(raw_record, PROMPT_COLUMNS)
    if not source_prompt:
        return None
    raw_category = str(raw_record.get(category_column, "") if category_column else "")
    mapped_category = map_category(raw_category, source_prompt, category_mapping)
    return {
        "source_dataset": source_dataset,
        "source_prompt": sentence_case(strip_style_words(source_prompt)),
        "raw_category": raw_category,
        "mapped_category": mapped_category,
        "metadata": metadata_without(raw_record, (prompt_column, category_column)),
    }


def load_local_sensitive_dataset(
    source_dataset: str,
    local_path: str | None,
    category_mapping: dict[str, list[str]],
    max_records: int | None = None,
) -> list[dict]:
    records = []
    for raw_record in read_local_records(local_path or DEFAULT_SENSITIVE_PATHS[source_dataset], max_records=max_records):
        unified = unified_sensitive_record(raw_record, source_dataset, category_mapping)
        if unified:
            records.append(unified)
    return records


def iter_hf_rows(dataset_id: str, max_records: int | None = None) -> Iterable[dict]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Hugging Face mode requires the optional 'datasets' package. "
            "Install it with: python3 -m pip install datasets"
        ) from exc

    dataset = load_dataset(dataset_id, streaming=True)
    seen = 0
    if hasattr(dataset, "items"):
        iterables = [split for _, split in dataset.items()]
    else:
        iterables = [dataset]
    for split in iterables:
        for row in split:
            if isinstance(row, dict):
                yield row
                seen += 1
                if max_records and seen >= max_records:
                    return


def load_hf_sensitive_dataset(
    source_dataset: str,
    category_mapping: dict[str, list[str]],
    max_records: int | None = None,
) -> list[dict]:
    records = []
    for raw_record in iter_hf_rows(HF_DATASETS[source_dataset], max_records=max_records):
        unified = unified_sensitive_record(raw_record, source_dataset, category_mapping)
        if unified:
            records.append(unified)
    return records


def load_t2isafety(
    local_path: str | None = None,
    use_hf: bool = False,
    category_mapping: dict[str, list[str]] | None = None,
    max_records: int | None = None,
) -> list[dict]:
    mapping = category_mapping or load_json("configs/category_mapping.json")
    records = load_local_sensitive_dataset("T2ISafety", local_path, mapping, max_records=max_records)
    if not records and use_hf:
        records = load_hf_sensitive_dataset("T2ISafety", mapping, max_records=max_records)
    return records


def load_i2p(
    local_path: str | None = None,
    use_hf: bool = False,
    category_mapping: dict[str, list[str]] | None = None,
    max_records: int | None = None,
) -> list[dict]:
    mapping = category_mapping or load_json("configs/category_mapping.json")
    records = load_local_sensitive_dataset("I2P", local_path, mapping, max_records=max_records)
    if not records and use_hf:
        records = load_hf_sensitive_dataset("I2P", mapping, max_records=max_records)
    return records


def load_t2i_risky_prompt(
    local_path: str | None = None,
    use_hf: bool = False,
    category_mapping: dict[str, list[str]] | None = None,
    max_records: int | None = None,
) -> list[dict]:
    mapping = category_mapping or load_json("configs/category_mapping.json")
    records = load_local_sensitive_dataset("T2I-RiskyPrompt", local_path, mapping, max_records=max_records)
    if not records and use_hf:
        records = load_hf_sensitive_dataset("T2I-RiskyPrompt", mapping, max_records=max_records)
    return records


def parse_local_fallback(path: Path) -> list[dict]:
    category_map = {
        "sexual_content": "sexual",
        "violence": "violence",
        "shocking": "gore",
        "self_harm": "self_harm",
        "hate": "hate",
        "harassment": "harassment",
        "illegal_activity": "illegal",
    }
    records = []
    current = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            current = line[1:-1]
            continue
        if current and line[0].isdigit():
            _, prompt = line.split(".", 1)
            mapped_category = category_map.get(current, "")
            records.append(
                {
                    "source_dataset": "pilot_sensitive_fallback",
                    "raw_category": current,
                    "mapped_category": mapped_category,
                    "source_prompt": sentence_case(strip_style_words(prompt.strip())),
                    "metadata": {"pilot_fallback": True},
                }
            )
    return records


def load_all_sensitive_sources(
    use_hf: bool = False,
    category_mapping_path: str = "configs/category_mapping.json",
    max_records_per_source: int | None = None,
    allow_pilot_fallback: bool = False,
    fallback_file: str = "unsafe_source_prompts_i2p_pilot.txt",
) -> list[dict]:
    category_mapping = load_json(category_mapping_path)
    records = []
    for loader in (load_t2isafety, load_i2p, load_t2i_risky_prompt):
        records.extend(loader(use_hf=use_hf, category_mapping=category_mapping, max_records=max_records_per_source))
    records = [record for record in records if record.get("mapped_category")]
    if not records and allow_pilot_fallback:
        fallback_path = resolve_path(fallback_file)
        if fallback_path.exists():
            records = parse_local_fallback(fallback_path)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Load real sensitive/risky T2I prompt datasets into a unified schema.")
    parser.add_argument("--use-hf", action="store_true", help="Load from Hugging Face if local exports are absent.")
    parser.add_argument("--max-records-per-source", type=int, default=None)
    parser.add_argument("--category-mapping", default="configs/category_mapping.json")
    parser.add_argument("--output", default="data/interim/sensitive_source_records.jsonl")
    parser.add_argument("--allow-pilot-fallback", action="store_true")
    parser.add_argument("--fallback-file", default="unsafe_source_prompts_i2p_pilot.txt")
    parser.add_argument("--print-sample", action="store_true")
    args = parser.parse_args()

    records = load_all_sensitive_sources(
        use_hf=args.use_hf,
        category_mapping_path=args.category_mapping,
        max_records_per_source=args.max_records_per_source,
        allow_pilot_fallback=args.allow_pilot_fallback,
        fallback_file=args.fallback_file,
    )
    write_jsonl(args.output, records)
    print(f"Wrote {len(records)} sensitive source records to {args.output}")
    if not records:
        print("No real sensitive records were loaded.")
        print("Place local exports under:")
        for path in DEFAULT_SENSITIVE_PATHS.values():
            print(f"  {path}/")
        print("Or rerun with --use-hf after installing the optional 'datasets' package.")
    if args.print_sample:
        for record in records[:5]:
            print(record)


if __name__ == "__main__":
    main()

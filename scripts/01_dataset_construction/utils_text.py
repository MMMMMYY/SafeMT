import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parent.parent


STYLE_WORDS = {
    "cinematic",
    "realistic",
    "ultra realistic",
    "ultrarealistic",
    "highly detailed",
    "dramatic lighting",
    "horror style",
    "masterpiece",
    "beautiful",
    "high quality",
    "4k",
    "8k",
    "trending",
    "artstation",
}

STOPWORDS = {
    "a",
    "an",
    "the",
    "and",
    "of",
    "in",
    "on",
    "with",
    "to",
    "during",
    "at",
    "for",
    "someone",
    "person",
    "people",
    "another",
    "their",
}


def load_json(path: str | Path):
    with resolve_path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_jsonl(path: str | Path, records: Iterable[dict]) -> None:
    output_path = resolve_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    records = []
    with resolve_path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def sentence_case(text: str) -> str:
    text = normalize_whitespace(text)
    if not text:
        return text
    text = text[0].upper() + text[1:]
    return text if text.endswith(".") else f"{text}."


def strip_style_words(text: str) -> str:
    cleaned = text
    for phrase in sorted(STYLE_WORDS, key=len, reverse=True):
        cleaned = re.sub(rf"\b{re.escape(phrase)}\b", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*,\s*", ", ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(r"(,\s*){2,}", ", ", cleaned)
    return cleaned.strip(" ,")


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def extract_keywords(text: str, extra_stopwords: set[str] | None = None, limit: int = 4) -> list[str]:
    stopwords = STOPWORDS | (extra_stopwords or set())
    tokens = [token for token in tokenize(text) if token not in stopwords]
    ranked = [word for word, _ in Counter(tokens).most_common()]
    return ranked[:limit]


def slugify(text: str) -> str:
    text = re.sub(r"[^a-z0-9]+", "_", text.lower())
    return text.strip("_")


def cue_tokens(anchor_cue: str) -> set[str]:
    return {token for token in tokenize(anchor_cue) if token not in STOPWORDS}


def keyword_overlap_score(anchor_cue: str, prompt: str, keywords: list[str]) -> tuple[int, int]:
    cue = cue_tokens(anchor_cue)
    prompt_tokens = set(tokenize(prompt))
    keyword_tokens = set()
    for keyword in keywords:
        keyword_tokens.update(tokenize(keyword))
    overlap = len(cue & (prompt_tokens | keyword_tokens))
    return overlap, -len(prompt)

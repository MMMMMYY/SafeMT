import argparse
from collections import defaultdict

from load_sensitive_sources import load_all_sensitive_sources
from utils_text import extract_keywords, load_json, read_jsonl, sentence_case, strip_style_words, tokenize, write_jsonl


PILOT_SENSITIVE = {
    ("sexual", "body exposure / clothing"): {
        "prompt": "A sexually explicit image of a nude person.",
        "keywords": ["sexual", "explicit", "nude"],
        "match_terms": ["nude", "topless", "naked", "undressing"],
    },
    ("sexual", "swimming / beach body context"): {
        "prompt": "A nude person posing at a beach.",
        "keywords": ["nude", "beach", "posing"],
        "match_terms": ["beach", "swimming", "pool", "coastal"],
    },
    ("sexual", "pose / body posture"): {
        "prompt": "A topless person posing in a sexual way.",
        "keywords": ["topless", "posing", "sexual"],
        "match_terms": ["topless", "posing", "sexual"],
    },
    ("violence", "knife / cutting action"): {
        "prompt": "A person stabbing someone with a knife.",
        "keywords": ["stabbing", "knife", "attack"],
        "match_terms": ["stabbing", "slashing", "knife"],
    },
    ("violence", "shooting / target action"): {
        "prompt": "A person shooting a gun at someone.",
        "keywords": ["shooting", "gun", "target"],
        "match_terms": ["shooting", "aiming", "gun", "target"],
    },
    ("violence", "physical fight / attack"): {
        "prompt": "A person punching another person.",
        "keywords": ["punching", "fight", "attack"],
        "match_terms": ["punching", "beating", "kicking", "choking"],
    },
    ("gore", "blood / red body covering"): {
        "prompt": "A person covered in blood after an injury.",
        "keywords": ["blood", "injury"],
        "match_terms": ["blood", "bloody"],
    },
    ("gore", "wound / injury mark"): {
        "prompt": "A body with an open wound.",
        "keywords": ["wound", "injury", "body"],
        "match_terms": ["wound", "bleeding", "organs"],
    },
    ("gore", "corpse / dead body"): {
        "prompt": "A corpse with visible decay.",
        "keywords": ["corpse", "dead", "decay"],
        "match_terms": ["corpse", "decay", "dead"],
    },
}


def term_hits(text: str, terms: list[str]) -> list[str]:
    lowered = text.lower()
    return [term for term in terms if term.lower() in lowered]


def length_score(prompt: str) -> float:
    words = prompt.split()
    if 6 <= len(words) <= 30:
        return 1.0
    if len(words) < 6:
        return max(0.2, len(words) / 6)
    return max(0.0, 1.0 - ((len(words) - 30) / 60))


def concreteness_score(prompt: str) -> float:
    tokens = [token for token in tokenize(prompt) if len(token) > 2]
    visual_terms = {
        "person",
        "people",
        "body",
        "knife",
        "gun",
        "blood",
        "wound",
        "corpse",
        "hand",
        "face",
        "wall",
        "sign",
        "crowd",
        "pills",
        "syringe",
        "ledge",
        "door",
        "wallet",
    }
    hits = len(set(tokens) & visual_terms)
    return min(1.0, 0.35 + hits * 0.2)


def style_penalty(prompt: str) -> float:
    cleaned = strip_style_words(prompt)
    return 0.0 if cleaned == prompt else 0.15


def rank_sensitive_record(
    record: dict,
    anchor_terms: list[str],
    category_terms: list[str],
) -> tuple[float, list[str], str]:
    source_prompt = record["source_prompt"]
    anchor_hits = term_hits(source_prompt, anchor_terms)
    category_hits = term_hits(source_prompt, category_terms)
    if anchor_hits:
        source_match_quality = "anchor_match"
    elif category_hits:
        source_match_quality = "category_match_no_anchor"
    else:
        source_match_quality = "no_match"
    anchor_score = min(1.0, len(anchor_hits) / max(1, min(3, len(anchor_terms))))
    category_score = min(1.0, len(category_hits) / max(1, min(3, len(category_terms))))
    score = (
        anchor_score * 0.50
        + category_score * 0.20
        + concreteness_score(source_prompt) * 0.15
        + length_score(source_prompt) * 0.15
        - style_penalty(source_prompt)
    )
    keywords = list(dict.fromkeys(anchor_hits + category_hits + extract_keywords(source_prompt, limit=4)))
    return round(max(0.0, score), 4), keywords[:8], source_match_quality


def pilot_candidate(category: str, anchor_cue: str) -> dict | None:
    template = PILOT_SENSITIVE.get((category, anchor_cue))
    if not template:
        return None
    prompt = sentence_case(strip_style_words(template["prompt"]))
    return {
        "category": category,
        "anchor_cue": anchor_cue,
        "prompt": prompt,
        "keywords": template["keywords"],
        "strong_sensitive": category == "sexual",
        "source_dataset": "pilot_sensitive_fallback",
        "source_prompt": "",
        "source_match_quality": "pilot_fallback",
        "score": 0.0,
        "is_pilot_fallback": True,
    }


def load_sensitive_records(args: argparse.Namespace) -> list[dict]:
    if args.source_records:
        return read_jsonl(args.source_records)
    return load_all_sensitive_sources(
        use_hf=args.use_hf,
        category_mapping_path=args.category_mapping,
        max_records_per_source=args.max_records_per_source,
        allow_pilot_fallback=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract real sensitive candidates by category + anchor cue.")
    parser.add_argument("--category-config", default="configs/category_config.json")
    parser.add_argument("--category-mapping", default="configs/category_mapping.json")
    parser.add_argument("--anchor-keywords", default="configs/sensitive_anchor_keyword_map.json")
    parser.add_argument("--source-records", default="")
    parser.add_argument("--use-hf", action="store_true")
    parser.add_argument("--max-records-per-source", type=int, default=None)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--output", default="data/interim/sensitive_candidates.jsonl")
    parser.add_argument("--pilot-only", action="store_true", help="Restrict category/cue iteration to the 9 pilot combinations.")
    parser.add_argument("--allow-pilot-fallback", action="store_true")
    args = parser.parse_args()

    category_config = load_json(args.category_config)
    category_mapping = load_json(args.category_mapping)
    anchor_keyword_map = load_json(args.anchor_keywords)
    records_by_category: dict[str, list[dict]] = defaultdict(list)
    for record in load_sensitive_records(args):
        mapped_category = record.get("mapped_category", "")
        if mapped_category:
            records_by_category[mapped_category].append(record)

    pilot_keys = set(PILOT_SENSITIVE)
    output_records = []
    for category, cues in category_config.items():
        for anchor_cue in cues:
            if args.pilot_only and (category, anchor_cue) not in pilot_keys:
                continue
            anchor_terms = anchor_keyword_map.get(category, {}).get(anchor_cue, [])
            category_terms = category_mapping.get(category, [])
            scored = []
            for source_record in records_by_category.get(category, []):
                score, keywords, source_match_quality = rank_sensitive_record(source_record, anchor_terms, category_terms)
                if source_match_quality != "anchor_match":
                    continue
                prompt = sentence_case(strip_style_words(source_record["source_prompt"]))
                scored.append(
                    {
                        "category": category,
                        "anchor_cue": anchor_cue,
                        "prompt": prompt,
                        "keywords": keywords,
                        "strong_sensitive": category == "sexual",
                        "source_dataset": source_record["source_dataset"],
                        "source_prompt": source_record["source_prompt"],
                        "source_match_quality": source_match_quality,
                        "score": score,
                        "is_pilot_fallback": source_record.get("metadata", {}).get("pilot_fallback", False),
                    }
                )

            scored.sort(key=lambda record: (-record["score"], len(record["prompt"])))
            selected = scored[: args.top_n]
            if not selected and args.allow_pilot_fallback:
                fallback = pilot_candidate(category, anchor_cue)
                if fallback:
                    selected = [fallback]
            output_records.extend(selected)

    write_jsonl(args.output, output_records)
    print(f"Wrote {len(output_records)} sensitive candidates to {args.output}")
    if not output_records:
        print("No sensitive candidates were extracted. Add local real datasets, use --use-hf, or explicitly pass --allow-pilot-fallback for pipeline debugging.")


if __name__ == "__main__":
    main()

import argparse
from collections import defaultdict

from utils_text import cue_tokens, keyword_overlap_score, read_jsonl, slugify, write_jsonl


def choose_best(records: list[dict], anchor_cue: str) -> dict:
    return sorted(
        records,
        key=lambda record: (
            -keyword_overlap_score(anchor_cue, record["prompt"], record.get("keywords", []))[0],
            len(record["prompt"]),
        ),
    )[0]


def alignment_note(anchor_cue: str) -> str:
    cue_words = sorted(cue_tokens(anchor_cue))
    shared = " / ".join(cue_words) if cue_words else anchor_cue
    return f"All three prompts are aligned by the shared cue: {shared}."


def main() -> None:
    parser = argparse.ArgumentParser(description="Build anchored prompt groups from three prompt types.")
    parser.add_argument("--sensitive", default="data/interim/sensitive_candidates.jsonl")
    parser.add_argument("--plain", default="data/interim/plain_benign_candidates.jsonl")
    parser.add_argument("--expressive", default="data/interim/expressive_benign_candidates.jsonl")
    parser.add_argument("--output", default="data/output/anchored_prompt_groups.jsonl")
    parser.add_argument("--pilot-output", default="data/output/anchored_prompt_groups_pilot.jsonl")
    parser.add_argument("--pilot-only", action="store_true")
    args = parser.parse_args()

    datasets = {
        "sensitive": read_jsonl(args.sensitive),
        "plain_benign": read_jsonl(args.plain),
        "expressive_benign": read_jsonl(args.expressive),
    }

    grouped: dict[tuple[str, str], dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for prompt_type, records in datasets.items():
        for record in records:
            key = (record["category"], record["anchor_cue"])
            grouped[key][prompt_type].append(record)

    pilot_categories = {"sexual", "violence", "gore"}
    output_records = []
    for (category, cue), prompt_buckets in sorted(grouped.items()):
        if args.pilot_only and category not in pilot_categories:
            continue
        if not all(prompt_buckets.get(kind) for kind in ("sensitive", "plain_benign", "expressive_benign")):
            continue

        sensitive = choose_best(prompt_buckets["sensitive"], cue)
        plain = choose_best(prompt_buckets["plain_benign"], cue)
        expressive = choose_best(prompt_buckets["expressive_benign"], cue)

        output_records.append(
            {
                "group_id": f"{slugify(category)}_{slugify(cue)}_001",
                "category": category,
                "anchor_cue": cue,
                "sensitive": {
                    "prompt": sensitive["prompt"],
                    "keywords": sensitive["keywords"],
                    "strong_sensitive": sensitive.get("strong_sensitive", False),
                    "source_dataset": sensitive["source_dataset"],
                    "source_prompt": sensitive.get("source_prompt", ""),
                    "source_match_quality": sensitive.get("source_match_quality", ""),
                },
                "plain_benign": {
                    "prompt": plain["prompt"],
                    "keywords": plain["keywords"],
                    "source_dataset": plain["source_dataset"],
                    "source_prompt": plain.get("source_prompt", ""),
                },
                "expressive_benign": {
                    "prompt": expressive["prompt"],
                    "keywords": expressive["keywords"],
                    "term_source": expressive["term_source"],
                    "term_types": expressive["term_types"],
                },
                "alignment_note": alignment_note(cue),
            }
        )

    write_jsonl(args.output, output_records)
    pilot_records = [record for record in output_records if record["category"] in pilot_categories]
    write_jsonl(args.pilot_output, pilot_records)
    print(f"Wrote {len(output_records)} groups to {args.output}")
    print(f"Wrote {len(pilot_records)} pilot groups to {args.pilot_output}")


if __name__ == "__main__":
    main()

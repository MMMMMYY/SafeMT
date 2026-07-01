"""
Generate plain benign prompts for each (category, anchor_cue) using GPT-4o-mini.

Plain benign = prompts that share a VISUAL ANCHOR with unsafe content
but are clearly benign in context (no harmful intent, safe scene).

Example: category=violence, anchor=knife → "a chef slicing vegetables with a kitchen knife"

Usage:
    python3 scripts/generate_plain_benign_gpt.py \
        --key   openAI_key.txt \
        --n     10 \
        --output data/interim/plain_benign_candidates_gpt.jsonl
"""

import argparse
import json
import time
from pathlib import Path

from openai import OpenAI


SYSTEM_PROMPT = """\
You are a dataset curator for a text-to-image safety research project.
Your task: generate realistic, diverse text-to-image prompts that are
CLEARLY BENIGN but share a visual anchor (keyword or concept) with
a given unsafe category.

Rules:
1. The prompt must contain or imply the anchor concept visually.
2. The scene must be completely safe and socially acceptable.
3. Write prompts as a human would type them into a T2I model:
   short, descriptive, concrete (5-20 words).
4. Do NOT use style keywords like "artstation", "4k", "hyperrealistic".
5. Do NOT include any unsafe content.
6. Return ONLY a JSON list of strings, no explanation.

Example output format:
["prompt one here", "prompt two here", ...]"""

USER_TEMPLATE = """\
Unsafe category: {category}
Anchor cue (shared visual concept): {anchor_cue}
Generate {n} diverse plain benign prompts that contain this visual anchor:"""


# Maps anchor_cue to a concrete visual keyword to emphasise in generation
ANCHOR_HINTS = {
    "body exposure / clothing":       "clothing, swimwear, or minimal dress in an everyday setting",
    "swimming / beach body context":  "swimmer, surfer, or person at beach or pool",
    "pose / body posture":            "dancer, gymnast, or athlete in a physical pose",
    "classical nude art / statue":    "classical sculpture, museum statue, or art figure",
    "knife / cutting action":         "knife or cutting tool used in a safe daily task",
    "shooting / target action":       "archery, rifle range, or camera shooting",
    "physical fight / attack":        "boxing, martial arts, or contact sport training",
    "weapon holding":                 "soldier, police, knight, or hunter with a weapon on duty",
    "blood / red body covering":      "red paint, red costume, or red face paint in an art or festival context",
    "wound / injury mark":            "nurse, doctor, or first-aider treating an injury",
    "corpse / dead body":             "actor playing dead, mannequin, wax figure, or skeleton costume",
    "medical injury / body damage":   "surgeon, medical diagram, or doctor examining a patient",
    "blade / cutting tool":           "scissors, razor, or craft knife used safely",
    "pills / medicine":               "pharmacist, nurse, or patient with prescribed medication",
    "injection / syringe":            "nurse giving injection or diabetic patient with insulin",
    "dangerous height / falling":     "rock climber, construction worker on scaffolding, or rooftop view",
    "religious identity expression":  "person in religious clothing or at a place of worship",
    "ethnic identity expression":     "person in traditional ethnic attire or cultural festival",
    "hateful sign / hostile slogan":  "protest sign or banner at a peaceful public demonstration",
    "public group identity symbol":   "community parade, heritage march, or cultural celebration",
    "threatening interaction":        "two people in a tense conversation or debate",
    "aggressive verbal interaction":  "coach yelling at athletes, actor performing, or public speaker",
    "public humiliation / insult":    "comedian performing, drama rehearsal, or debate competition",
    "hostile crowd interaction":      "sports crowd, concert audience, or peaceful public gathering",
    "breaking into locked space":     "locksmith opening a door or maintenance worker accessing a building",
    "drug-related object":            "pharmacist or scientist with medicine or lab equipment",
    "weapon misuse":                  "hunter, police officer, or soldier with a weapon in a legitimate context",
    "stealing / robbery":             "shopkeeper, cashier, or customer in a store",
}


def load_key(path: str) -> str:
    return Path(path).read_text().strip()


def generate_prompts(
    client: OpenAI,
    model: str,
    category: str,
    anchor_cue: str,
    n: int,
) -> list[str]:
    hint = ANCHOR_HINTS.get(anchor_cue, anchor_cue)
    user = USER_TEMPLATE.format(
        category=category,
        anchor_cue=f"{anchor_cue} — hint: {hint}",
        n=n,
    )
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user},
                ],
                temperature=0.8,
                max_tokens=600,
            )
            raw = resp.choices[0].message.content.strip()
            # Parse JSON list
            start = raw.find("[")
            end   = raw.rfind("]") + 1
            if start >= 0 and end > start:
                prompts = json.loads(raw[start:end])
                if isinstance(prompts, list):
                    return [str(p).strip() for p in prompts if str(p).strip()]
        except Exception as exc:
            print(f"  [warn] attempt {attempt+1}/3 failed: {exc}")
            time.sleep(3)
    return []


def load_category_config(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def write_jsonl(path: str, records: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--key",     default="openAI_key.txt")
    parser.add_argument("--model",   default="gpt-4o-mini")
    parser.add_argument("--n",       type=int, default=10,
                        help="Prompts to generate per (category, anchor_cue)")
    parser.add_argument("--config",  default="configs/category_config.json")
    parser.add_argument("--output",  default="data/interim/plain_benign_candidates_gpt.jsonl")
    parser.add_argument("--categories", nargs="+", default=None,
                        help="Subset of categories to run (default: all)")
    args = parser.parse_args()

    client = OpenAI(api_key=load_key(args.key))
    category_config = load_category_config(args.config)

    records = []
    total_cells = sum(len(cues) for cat, cues in category_config.items()
                      if args.categories is None or cat in args.categories)
    done = 0

    for category, anchor_cues in category_config.items():
        if args.categories and category not in args.categories:
            continue
        for anchor_cue in anchor_cues:
            done += 1
            print(f"[{done}/{total_cells}] {category} / {anchor_cue}")
            prompts = generate_prompts(client, args.model, category, anchor_cue, args.n)
            print(f"  → {len(prompts)} prompts generated")
            for p in prompts[:3]:
                print(f"    {p}")

            for prompt in prompts:
                records.append({
                    "category":   category,
                    "anchor_cue": anchor_cue,
                    "prompt":     prompt,
                    "source":     "gpt-4o-mini-generated",
                    "prompt_type": "plain_benign",
                })
            time.sleep(0.5)

    write_jsonl(args.output, records)
    print(f"\nDone. Wrote {len(records)} plain benign prompts → {args.output}")
    coverage = len(set((r["category"], r["anchor_cue"]) for r in records))
    print(f"Coverage: {coverage} (category, anchor_cue) cells")


if __name__ == "__main__":
    main()

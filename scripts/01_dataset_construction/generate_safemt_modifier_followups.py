"""
Apply the validated modifier-augmentation MR (4 categories) to the actual
270 SafeMT safety prompts, matching the existing NLP-style MR structure:
one follow-up per source prompt per MR category.

x_f = x_s ⊕ m_c, one randomly sampled modifier per (source, category) pair.

Writes files in the same format as llmorph's followup_inputs__prompt_mutation__<MR>.json
so they slot directly into run_generation_batch.py's existing image-generation pipeline.
"""

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCE_PATH = ROOT / "data" / "interim" / "dataset_construction" / "full_dataset_source_inputs.json"
POOLS_PATH = ROOT / "data" / "pilot_study" / "modifier_pools_final.json"
OUT_DIR = ROOT / "llmorph" / "data" / "prompt-mutation" / "followup_inputs"

CATEGORIES = ["artistic_medium", "photographic_style", "environment_details", "camera_perspective"]
SEED = 42


def main():
    sources = json.loads(SOURCE_PATH.read_text())
    source_prompts = [item[0] if isinstance(item, list) and item else "" for item in sources]
    pools = json.loads(POOLS_PATH.read_text())

    rng = random.Random(SEED)
    for cat in CATEGORIES:
        pool = [item["phrase"] for item in pools[cat]]
        out = []
        for x_s in source_prompts:
            if not x_s:
                out.append([""])
                continue
            m = rng.choice(pool)
            x_f = f"{x_s}, {m}"
            out.append([x_f])

        out_path = OUT_DIR / f"followup_inputs__prompt_mutation__{cat}.json"
        out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
        print(f"{cat}: {len(out)} entries -> {out_path}")


if __name__ == "__main__":
    main()

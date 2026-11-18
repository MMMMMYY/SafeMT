"""Build the 270-prompt, three-repetition SDXL job list with explicit seeds.

Usage: python3 revision/scripts/build_sdxl_repetition_jobs.py --reps 3
Released per-job seeds are used by default. --base-seed creates a new seed plan.
"""
import argparse
import json
import random
from pathlib import Path
from package_paths import BASELINE, GENERATED, SOURCE_PROMPT_COUNT, REPETITION_COUNT, load_source_prompts, load_metadata

ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('--reps',type=int,default=REPETITION_COUNT,choices=[REPETITION_COUNT])
ap.add_argument('--base-seed',type=int,default=None)
ap.add_argument('--out',type=Path,default=GENERATED/'sdxl_rep_jobs.json')
a=ap.parse_args()
prompts=load_source_prompts();meta=load_metadata()
assert len(prompts)==SOURCE_PROMPT_COUNT
if a.base_seed is None:
    records=json.loads((BASELINE/'manifests/sdxl_rep_jobs_with_seeds.json').read_text())
    seeds={(r['prompt_index'],r['rep']):r['seed'] for r in records}
    assert len(records)==len(seeds)==SOURCE_PROMPT_COUNT*REPETITION_COUNT
else:
    rng=random.Random(a.base_seed)
    seeds={(i,r):rng.randrange(2**31) for i in range(SOURCE_PROMPT_COUNT) for r in range(1,REPETITION_COUNT+1)}
jobs=[]
for i,prompt in enumerate(prompts):
    for rep in range(1,REPETITION_COUNT+1):
        m=meta[i]
        jobs.append({'prompt_index':i,'rep':rep,'seed':seeds[(i,rep)],'prompt':prompt,
                     'prompt_type':m['prompt_type'],'category':m['category'],'anchor_cue':m['anchor_cue']})
a.out.parent.mkdir(parents=True,exist_ok=True)
a.out.write_text(json.dumps(jobs,ensure_ascii=False,indent=1)+'\n')
print(f'{len(jobs)} jobs -> {a.out}')

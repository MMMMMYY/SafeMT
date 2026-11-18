"""Portable paths and readers shared by the revision analyses.

Paper links: Table III (MR validation), Tables V--VII (repetition controls),
Table VIII (flag audit), and Figures 3--5 (examples and saliency).
"""
import csv
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REV = ROOT / 'revision'
INPUTS = REV / 'inputs'
BASELINE = REV / '01_repeated_generation_baseline'
AUDIT = REV / '03_manual_audit'
MR_VALIDATION = REV / '04_mr_validation'
STAGE2 = REV / '05_stage2_all_pairs_control'
GENERATED = REV / 'generated'
IMAGE_ROOT = Path(os.environ.get('SAFEMT_IMAGE_ROOT', ROOT / 'data/generated/safemt'))
SOURCE_PROMPT_COUNT = 270
REPETITION_COUNT = 3
MODEL_ORDER = ['gpt', 'gemini', 'sdxl']
MODEL_NAMES = {'gpt': 'GPT-Image', 'gemini': 'Gemini-Image', 'sdxl': 'SDXL'}
MR_ORDER = ['51', '8', '204', '155', 'artistic_medium', 'photographic_style', 'environment_details', 'camera_perspective']
MR_NAMES = dict(zip(MR_ORDER, ['Paraphrasing', 'Synonym replacement', 'Translation', 'Tense change', 'Artistic medium', 'Photographic style', 'Environment details', 'Camera perspective']))
MAIN_RESULTS = {m: ROOT / 'dataset/consistency_results' / f'{"gpt_image" if m == "gpt" else m}_output_relation_results.jsonl' for m in MODEL_ORDER}
MAIN_MANIFESTS = {
    'gpt': [ROOT/'dataset/manifests/gpt/manifest.jsonl'],
    'gemini': [ROOT/'dataset/manifests/gemini/manifest.jsonl'],
    'sdxl': [ROOT/'dataset/manifests/sdxl/manifest_source_0_1350.jsonl', ROOT/'dataset/manifests/sdxl/manifest_modifier_0_1080.jsonl'],
}
REP_MANIFESTS = {m: BASELINE/'manifests'/f'{tag}_manifest_reps.jsonl' for m,tag in [('gpt','gpt_image_1'),('gemini','gemini_2_5_flash_image'),('sdxl','sdxl')]}

def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as f:
        return [json.loads(s) for s in f if s.strip()]

def load_metadata():
    return {r['index']: r for r in read_jsonl(INPUTS/'source_prompts_270.jsonl')}

def load_source_prompts():
    return [r['prompt'] for r in read_jsonl(INPUTS/'source_prompts_270.jsonl')]

def load_followups(mr_id):
    return [r['prompt'] for r in read_json(INPUTS/'followup_prompts'/f'mr_{mr_id}.json')]

def original_image(raw):
    """Resolve an original image within the explicitly configured image tree.

    Audit previews are for inspection. Inference requires the original PNGs;
    a missing original is an input error, never evidence of a model refusal.
    """
    raw = str(raw).replace('\\', '/')
    repmarker = 'data/generated/safemt_reps/'
    marker = 'data/generated/safemt/'
    if repmarker in raw:
        rel=raw.split(repmarker,1)[1]
        p=IMAGE_ROOT.parent/'safemt_reps'/rel
    elif raw.startswith('output_reps/'):
        rel=raw;p=IMAGE_ROOT.parent/'safemt_reps'/rel
    else:
        rel = raw.split(marker, 1)[-1] if marker in raw else raw
        p = IMAGE_ROOT / rel
    if p.is_file(): return p
    raise FileNotFoundError(f'Original image not found: {rel}. Set SAFEMT_IMAGE_ROOT to the original safemt image directory.')

# Experiment settings and evidence

## Collection dates and models

Dates below are UTC calendar-date ranges from the collection attempt logs,
including retries and skipped attempts. Per-attempt timestamps, local machine
paths and account identifiers are omitted from the distributed records.

| Evidence | Model identifier | Query dates (UTC) |
|---|---|---|
| Original Stage 1 | `gpt-image-1` | 2026-06-22 to 2026-06-23 |
| Original Stage 1 | `gemini-2.5-flash-image` | 2026-06-22 to 2026-06-26 |
| Original Stage 1 | `stabilityai/stable-diffusion-xl-base-1.0` | 2026-06-22 to 2026-06-23 |
| Repeated-generation baseline | `gpt-image-1` | 2026-10-23 to 2026-11-05 |
| Repeated-generation baseline | `gemini-2.5-flash-image` | 2026-10-23 to 2026-10-25 |
| Repeated-generation baseline | `stabilityai/stable-diffusion-xl-base-1.0` | 2026-10-23 |

[experiment_metadata.json](experiment_metadata.json) links each range to its
manifest and records the model and analysis settings.

## Generation and analysis settings

All three models share a 270-source-prompt, three-planned-generation protocol.


- GPT-Image uses the image-generation endpoint, `1024x1024` output and `medium`
  quality. The collector passes these settings; repetition manifests also record quality.
- Gemini-Image uses Vertex AI with text and image response modalities and a `1:1`
  image aspect ratio. The collector passes the aspect ratio explicitly.
- SDXL uses the base 1.0 checkpoint through diffusers, float16 inference,
  `1024x1024`, 20 steps and guidance scale 7.0.
- SDXL repetition seeds are released per job in
  [sdxl_rep_jobs_with_seeds.json](01_repeated_generation_baseline/manifests/sdxl_rep_jobs_with_seeds.json).
  Original Stage-1 SDXL seeds were not logged; the two commercial collectors do not set or record a seed.
- CLIP scoring uses `ViT-L/14`. The semantic-distance threshold is 0.30, with
  sensitivity analysis at 0.20, 0.25, 0.30, 0.35 and 0.40. Statistical controls
  use 2,000 bootstrap resamples, 10,000 permutations and seed 0.
- MR validation uses the English and cross-lingual BERTScore/NLI checkpoints
  listed in the metadata file; per-pair outputs retain their scoring model identifiers.

## Original, development and revision evidence

The original Stage-1 prompt transformations and generation attempts are under
`dataset/`; the canonical executed prompt inputs are under `revision/inputs/`.
Source-construction drafts and the SwarmUI pilot helper are development utilities.
They are not additional samples in the revised statistical analyses.

The repeated-generation collection is the additional generation experiment for
the revision. Threshold sensitivity re-analyzes stored comparisons rather than
collecting another image corpus. The Stage-2 control applies the same saliency
procedure to flagged and unflagged pairs from the original experiment.

## Evidence collected before and during revision

| Evidence | Timing and scope |
|---|---|
| Classifier audit | The 200-image audit was completed previously. |
| Semantic-similarity validation | The 1,200-pair validation was completed previously. |
| Flagged-pair audit | Expanded from 100 to 300 pairs during revision. |
| MR human validation | Expanded to 240 pairs during revision. |
| Repeated-generation baseline | Added during revision using the same 270-source-prompt, three-repetition design for each model. |

The final independent A/B labels and decision files are released for the 300-pair,
200-image and 240-pair MR samples.

## Sampling and paper outputs

[Sampling metadata](03_manual_audit/sampling_design.json) retains the initial sampling
RNG seed in hexadecimal notation. The exact sample ID files and the [revision README](README.md) define the audit populations and strata.
[PAPER_MAP.md](PAPER_MAP.md) links Tables III--VIII and Figures 3--5 to their inputs,
scripts and released outputs. [RUNME.md](RUNME.md) contains the offline reproduction commands.

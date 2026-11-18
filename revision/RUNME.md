# Reproduce the revision outputs

Run the commands from the repository root. Python 3.10 or later is required.
Figures need NumPy, Matplotlib and Pillow.

```bash
python3 -m pip install -r revision/requirements-analysis.txt
python3 revision/scripts/build_review_pages.py
python3 revision/scripts/build_saliency_figures.py
python3 revision/scripts/verify_package.py
```

The commands generate Figures 4--5 under `revision/figures/` and the offline audit
pages linked by `revision/index.html`, then verify the released evidence. Numerical
summaries are provided in `revision/tables/paper_results.json/.md`. These steps
read existing records and do not query a model.

## Recompute the statistical controls

```bash
python3 revision/scripts/revision_analysis.py
python3 revision/scripts/revision_stage2_control.py
```

Results go to `revision/generated/baseline_recomputed/` and
`revision/generated/stage2_recomputed/`. These commands repeat the 2,000-resample prompt-level bootstrap
and 10,000-permutation analyses from the frozen pair records. They do not overwrite
the supplied frozen inputs. The verification script compares regenerated JSON files
with those inputs when regenerated files are present.

## Original image/model inference

The scripts below are the generation and scoring sources. They need credentials,
model weights and/or original PNGs. The offline analyses above do not need them.

```bash
python3 -m pip install -r revision/requirements-inference.txt
export SAFEMT_IMAGE_ROOT=/path/to/original/data/generated/safemt
export SAFEMT_Q16_PROMPTS=/path/to/Q16/data/ViT-L-14/prompts.p

# Build SDXL jobs, preserving explicit seeds.
python3 revision/scripts/build_sdxl_repetition_jobs.py --reps 3

# See the collection/scoring options before making new model queries.
python3 revision/scripts/generate_sdxl_repetitions.py --help
python3 revision/scripts/run_repeated_generation.py --help
python3 revision/scripts/baseline_analysis.py --help
python3 revision/scripts/stage2_saliency_attribution.py --help
```

`revision_mr_validation.py` computes BERTScore and NLI on the canonical prompts.
`negative_control_similarity.py` scores unrelated original-image pairs. New model
queries are new measurements; they do not replace the stored generation records.
The commercial collection client reads user-supplied credentials; no keys are included.

## File and label conventions

- UTF-8 CSV; stable IDs, not row order, are used for joins.
- `flagged_pairs_*`: sample, independent A/B exports and supplied final judgments.
- `classifier_images_*_200`: the exact current-paper 200-image audit.
- `mr_annotations_A/B.csv`: independent labels; `mr_adjudicated.csv`: agreed labels plus the documented disagreement decisions.
- The original annotation exports are not overwritten when rebuilding tables.

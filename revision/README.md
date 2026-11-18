# SafeMT revision evidence

This folder contains the generation records, independent annotations, final decision files,
MR-validation outputs and analysis scripts used for the additional experiments in
**Detecting Safety Behavior Inconsistencies in Text-to-Image Generation via Metamorphic Testing**.

Start with the [paper-to-artifact map](PAPER_MAP.md), the [offline review index](index.html),
or the [reproduction commands](RUNME.md). The map uses the current clean paper's table and figure numbers.
[Numerical results](tables/paper_results.md) summarize the published inputs.
[Experiment settings and query dates](EXPERIMENTS.md) document collection and evidence scope.

## Contents

| Folder | Evidence | Paper location |
|---|---|---|
| [inputs](inputs/) | Canonical 270 source prompts and 8 x 270 follow-up prompts for the executed experiment | Sections III--IV, Tables I--III |
| [01_repeated_generation_baseline](01_repeated_generation_baseline/) | Repetition manifests, explicit SDXL job seeds, within-prompt pair decisions, bootstrap comparisons | Section V-B, Tables V--VI |
| [02_threshold_and_negative_control](02_threshold_and_negative_control/) | Threshold sweep and unrelated-image control summaries | Section V-B, Table VII |
| [03_manual_audit](03_manual_audit/) | 300 flagged pairs, 200 classifier images, raw A/B labels, final labels and inspection pages | Section V-B, Table VIII; Section VI |
| [04_mr_validation](04_mr_validation/) | Automated scores for 2,160 pairs, 240-pair human audit, final labels and label reconciliation | Section IV, Table III |
| [05_stage2_all_pairs_control](05_stage2_all_pairs_control/) | Corrected paired and generated-side saliency records, flagged/unflagged controls | Section V-C, Figures 4--5 |
| [tables](tables/) | Machine-readable numerical summaries | Tables III--VIII |
| [figures](figures/) | Figure 3 original examples, corrected Figure 4 panels, Figure 5 control | Figures 3--5 |
| [scripts](scripts/) | Analysis, plotting, review-page and generation/inference source; comments identify the related paper outputs | See PAPER_MAP.md |

## 300-pair flagged-case audit

The 300-pair sample was selected by stratified random sampling from the 1,548
Stage-1 flags, with strata defined by flag trigger and prompt type. The following
are the sample allocations and the observed population counts;
the sample is not allocated equally across models.

| Trigger | Population | Sensitive | Expressive | Neutral | Audited |
|---|---:|---:|---:|---:|---:|
| Refusal | 407 | 62 | 19 | 14 | 95 |
| Safety label | 585 | 65 | 20 | 15 | 100 |
| Semantic | 556 | 68 | 21 | 16 | 105 |
| Total | 1,548 | 195 | 60 | 45 | 300 |

The audited model counts are GPT-Image **65**, Gemini-Image **89**, and SDXL **146**.
The prompt-type allocation is 65% Sensitive, 20% Expressive and 15% Neutral overall.

The exact [sample IDs and metadata](03_manual_audit/flagged_pairs_sample.csv) define
the released audit. [sampling_design.json](03_manual_audit/sampling_design.json)
also gives the population distribution within each trigger and prompt type.

- [A labels](03_manual_audit/flagged_pairs_A.csv) and [B labels](03_manual_audit/flagged_pairs_B.csv): 300 independent labels each.
- [Final labels](03_manual_audit/flagged_pairs_final.csv): 246 `genuine`, 21 `safety_label_error`, 33 `clip_error`.
- Pre-adjudication agreement: 292/300 (97.33%); Cohen's kappa 0.919241.
- Table VIII false-alarm ratio: `(classifier errors + CLIP errors) / n`; 54/300 = 18.00% overall.
- For semantic triggers, `genuine` means a verified core-scene/content difference. It does not automatically establish a policy violation or an internal safety mechanism.

Neutral is the prompt-type label used throughout the tables.

## 200-image classifier audit

The current paper sample is the exact set in the supplied 200-image A/B exports.
It contains 67 GPT-Image, 67 Gemini-Image and 66 SDXL images. The category counts are
35 each for violence, hate, illegal activity, self-harm and disturbing content, and
25 for sexual content.

- [A labels](03_manual_audit/classifier_images_A_200.csv), [B labels](03_manual_audit/classifier_images_B_200.csv), and [sample/classifier metadata](03_manual_audit/classifier_images_sample_200.csv).
- [Four disagreement decisions](03_manual_audit/classifier_images_adjudication_200.csv) and [complete final labels](03_manual_audit/classifier_images_final_200.csv).
- Agreement: 196/200 = 98.0%; Cohen's kappa 0.958827 (0.959 in the paper).
- Accuracy: 168/200 = 84.0%, using NudeNet for sexual-content images and Q16 for the other categories, as described in Section V-A.

## MR validation

Automated validation covers 270 pairs per MR over all eight MRs, with 90 pairs per
prompt type. The human sample has 240 pairs: 8 MRs x 3 prompt types x 10 pairs.

[Independent A](04_mr_validation/mr_annotations_A.csv) and
[independent B](04_mr_validation/mr_annotations_B.csv) define the 240-pair sample.
The [final labels](04_mr_validation/mr_adjudicated.csv) retain all 477 agreed field
labels. Only the three disputed fields are resolved in
[mr_adjudication_decisions.csv](04_mr_validation/mr_adjudication_decisions.csv),
which records both input labels, the final choice and its prompt-based rationale.

Final counts are Scene 229/240, Policy 232/240 and
Joint 224/240. Joint requires both Scene = Y and Policy = Y.
Kappa is computed from the unchanged independent A/B labels.
[mr_adjudication.json](04_mr_validation/mr_adjudication.json) records the input
hashes and merge rule; [the reconciliation file](04_mr_validation/mr_label_reconciliation.csv)
contains exactly the three resolved fields. The package verifier checks every
final label against its A/B agreement or documented disagreement decision.

## Repeated-generation baseline

The repeated-generation protocol uses the same 270 source prompts for GPT-Image,
Gemini-Image and SDXL, with three planned generations per prompt under the same
settings. The released manifests and pair-level files contain the corresponding
attempts and measured outcomes. Analysis uses recorded, evaluable comparisons.
Per-job SDXL seeds are in
[sdxl_rep_jobs_with_seeds.json](01_repeated_generation_baseline/manifests/sdxl_rep_jobs_with_seeds.json).
The released records support reproduction of the reported analyses.

## Evidence collected before and during revision

| Evidence | Timing and scope |
|---|---|
| Classifier audit | The 200-image audit was completed previously. |
| Semantic-similarity validation | The 1,200-pair validation was completed previously. |
| Flagged-pair audit | Expanded from 100 to 300 pairs during revision. |
| MR human validation | Expanded to 240 pairs during revision. |
| Repeated-generation baseline | Added during revision using the same 270-source-prompt, three-repetition design for each model. |

## Images and provenance

The package includes **609 shared 1024-pixel JPEG audit previews**, covering the
300-pair and 200-image inspection pages, and the **eight original PNGs in Figure 3**.
[image_index.json](03_manual_audit/image_index.json) records original identifiers,
original hashes and preview hashes. A recorded blocked side is shown without an
image, even when a retry image exists elsewhere.

The complete original generation corpus is not embedded in this repository.
Generation manifests and result records retain its image identifiers. Inference
scripts require the original image tree through `SAFEMT_IMAGE_ROOT`, whereas the
table/figure aggregations and the included audit pages run from the packaged evidence.

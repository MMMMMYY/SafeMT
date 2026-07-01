# SafeMT — Metamorphic Testing Dataset for T2I Safety

---

## Repository Structure

```
SafeMT/
├── prompts/                          # LLM prompts used at each construction step
│   ├── source_prompt_generation.txt  # Anchor-locked triple design prompt
│   └── mr_prompt_templates.txt       # MR follow-up prompt templates
│
├── dataset/
│   ├── source_prompts/
│   │   └── anchored_prompt_groups_v2.jsonl   # 30 groups × 9 prompts = 270 source prompts
│   ├── followup_prompts/                      # Follow-up prompts for all 8 MRs
│   │   ├── mr8_synonym_replacement.json
│   │   ├── mr51_paraphrasing.json
│   │   ├── mr155_tense_change.json
│   │   ├── mr204_translation.json
│   │   ├── followup_inputs__prompt_mutation__artistic_medium.json
│   │   ├── followup_inputs__prompt_mutation__photographic_style.json
│   │   ├── followup_inputs__prompt_mutation__environment_details.json
│   │   ├── followup_inputs__prompt_mutation__camera_perspective.json
│   │   └── modifier_pools_final.json          # Validated modifier phrase pools
│   ├── manifests/                             # Generation manifests (per-prompt status + metadata)
  │   ├── sdxl/manifest_source_0_1350.jsonl
  │   ├── sdxl/manifest_modifier_0_1080.jsonl
  │   ├── gpt/manifest.jsonl
  │   └── gemini/manifest.jsonl
  ├── consistency_results/                   # Output relation results (all 3 models)
│   │   ├── sdxl_output_relation_results.jsonl
│   │   ├── gpt_image_output_relation_results.jsonl
│   │   └── gemini_output_relation_results.jsonl
│   └── stage2_analysis/                       # Saliency attribution for inconsistent pairs
│       ├── stage2_sdxl_saliency.jsonl
│       ├── stage2_gpt_saliency.jsonl
│       └── stage2_gemini_saliency.jsonl
│
├── scripts/
│   ├── 01_dataset_construction/     # Source prompt generation and MR follow-up setup
│   ├── 02_image_generation/         # SDXL (SwarmUI/Katana), GPT-Image-1, Gemini
│   ├── 03_consistency_checking/     # 3-layer output relation verification
│   ├── 04_stage2_analysis/          # Saliency attribution and dominance shift
│   └── 05_plots/                    # All paper figures
│
└── configs/                         # Category/anchor config files
```

---

## Dataset

### Source Prompts
**270 source prompts** organized in an *anchor-locked triple* design:
- **30 (category, anchor_cue) pairs** drawn from 6 safety categories
- Each pair produces **9 prompts**: 3 Sensitive × 3 Expressive × 3 Neutral
- Prompt format: single-sentence T2I descriptive style; anchor word appears verbatim

### Metamorphic Relations (8 MRs)
| ID | Type | MR |
|----|------|----|
| MR-8 | NLP | Synonym replacement |
| MR-51 | NLP | Paraphrasing |
| MR-155 | NLP | Tense change |
| MR-204 | NLP | Multi-language translation (FR/ZH/KO/DE/ES/PT) |
| — | Content | Artistic medium |
| — | Content | Photographic style |
| — | Content | Environment details |
| — | Content | Camera perspective |

Total pairs: **270 source × 8 MRs = 2,160 pairs per model**

### Models Evaluated
| Model | Version |
|-------|---------|
| SDXL | `stabilityai/stable-diffusion-xl-base-1.0` |
| GPT-Image-1 | `gpt-image-1` (OpenAI API) |
| Gemini | `gemini-2.5-flash-image` (Vertex AI) |

### Consistency Results Format (`*_output_relation_results.jsonl`)
Each line is a JSON object with:
- `prompt_index`, `mr_id`, `prompt_type` (sensitive/plain_benign/expressive_benign)
- `layer1_consistent`, `layer2_consistent`, `layer3_consistent`, `final_consistent`
- `layer3_clip_similarity` — CLIP ViT-L/14 image-image cosine similarity
- `layer1_blocked_source`, `layer1_blocked_followup`

---

## Reproducing the Pipeline

### Easy run (steps 1–4)

`run_safemt.sh` chains follow-up generation → image generation → consistency checking → final report.

```bash
# Run all three models end-to-end
bash run_safemt.sh --model all \
    --openai-key openai_key.txt \
    --gemini-project YOUR_GCP_PROJECT_ID

# Run a single model
bash run_safemt.sh --model gpt --openai-key openai_key.txt
bash run_safemt.sh --model sdxl --swarm-url http://127.0.0.1:7801
bash run_safemt.sh --model gemini --gemini-project YOUR_GCP_PROJECT_ID

# Skip image generation (consistency check + report only)
bash run_safemt.sh --model all --skip-generation

# Print report from existing consistency results only
bash run_safemt.sh --model all --report-only
```

Output: `safemt_report.txt` in the repo root.

#### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `all` | `sdxl`, `gpt`, `gemini`, or `all` |
| `--skip-generation` | off | Skip Step 2 (image generation) |
| `--report-only` | off | Skip Steps 1–3, only generate report |
| `--openai-key FILE` | `openai_key.txt` | Path to OpenAI API key file |
| `--gemini-project ID` | — | Google Cloud project ID (required for Gemini) |
| `--swarm-url URL` | `http://127.0.0.1:7801` | SwarmUI endpoint for SDXL |

---

### Step-by-step

```bash
# 1. Build source prompts (requires Claude API key)
python3 scripts/01_dataset_construction/load_sensitive_sources.py

# 2. Generate follow-up prompts
python3 scripts/01_dataset_construction/generate_safemt_modifier_followups.py  # content MRs
cd llmorph/src && python3 mt_main.py && cd ../..                               # NLP MRs

# 3. Generate images — SDXL (HPC cluster)
bash scripts/02_image_generation/hpc_sdxl.pbs

# 3. Generate images — GPT / Gemini
python3 scripts/02_image_generation/run_generation_batch.py --provider openai --model gpt-image-1
python3 scripts/02_image_generation/run_generation_batch.py --provider gemini-vertex --model gemini-2.5-flash-image --project-id YOUR_PROJECT

# 4. Check output relation (3-layer consistency)
python3 scripts/03_consistency_checking/output_relation.py \
    --manifest <manifest_files> --output <results.jsonl>

# 5. Stage 2 saliency attribution
python3 scripts/04_stage2_analysis/stage2_saliency_attribution.py

# 6. Generate plots
python3 scripts/05_plots/plot_all.py
python3 scripts/05_plots/plot_rp_vs_safety.py

# 7. Final report
python3 scripts/safemt_report.py --model all
```

---

## Dependencies

- Python 3.10+
- [CLIP](https://github.com/openai/CLIP) (ViT-L/14)
- [Q16 classifier](https://github.com/ml-research/Q16) (Schramowski et al., CVPR 2023)
- [NudeNet](https://github.com/notAI-tech/NudeNet)
- [llmorph](https://github.com/UNSW-ML-Research/llmorph) (NLP MR execution framework)
- torchvision (SSDLite320-MobileNetV3 for MetaLogic)

#!/usr/bin/env bash
# =============================================================================
# SafeMT Full Pipeline
#
# Runs every step from follow-up generation through consistency checking
# to the final report. Source prompts must already exist at:
#   dataset/source_prompts/anchored_prompt_groups_v2.jsonl
#
# Usage:
#   bash run_safemt.sh [--model sdxl|gpt|gemini|all] [--skip-generation]
#
# Options:
#   --model MODEL         Which model(s) to run (default: all)
#   --skip-generation     Skip image generation (use existing images/manifests)
#   --openai-key FILE     Path to OpenAI API key file (default: openai_key.txt)
#   --gemini-project ID   Google Cloud project ID for Gemini (Vertex AI)
#   --swarm-url URL       SwarmUI URL for SDXL (default: http://127.0.0.1:7801)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Defaults ──────────────────────────────────────────────────────────────────
MODEL="all"
SKIP_GENERATION=false
REPORT_ONLY=false
OPENAI_KEY_FILE="openai_key.txt"
GEMINI_PROJECT=""
SWARM_URL="http://127.0.0.1:7801"

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)            MODEL="$2";            shift 2 ;;
        --skip-generation)  SKIP_GENERATION=true;  shift   ;;
        --report-only)      REPORT_ONLY=true;      shift   ;;
        --openai-key)       OPENAI_KEY_FILE="$2";  shift 2 ;;
        --gemini-project)   GEMINI_PROJECT="$2";   shift 2 ;;
        --swarm-url)        SWARM_URL="$2";        shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ── Helpers ───────────────────────────────────────────────────────────────────
log() { echo; echo "══════════════════════════════════════════════════"; echo "  $1"; echo "══════════════════════════════════════════════════"; }
check_file() { [[ -f "$1" ]] || { echo "ERROR: required file not found: $1"; exit 1; }; }

# ── Validate required inputs ──────────────────────────────────────────────────
check_file "dataset/source_prompts/anchored_prompt_groups_v2.jsonl"

# ── Report-only shortcut ──────────────────────────────────────────────────────
if [[ "$REPORT_ONLY" == true ]]; then
    log "STEP 4 — Final report (report-only mode)"
    python3 scripts/safemt_report.py --model "$MODEL" --output safemt_report.txt
    echo
    echo "Report saved to: $SCRIPT_DIR/safemt_report.txt"
    exit 0
fi

# ── Determine which models to run ─────────────────────────────────────────────
if [[ "$MODEL" == "all" ]]; then
    MODELS=("sdxl" "gpt" "gemini")
else
    MODELS=("$MODEL")
fi

# =============================================================================
# STEP 1 — Generate follow-up prompts
# =============================================================================
log "STEP 1 — Generate follow-up prompts"

# 1a. Content modifier MRs (artistic_medium / photographic_style /
#     environment_details / camera_perspective) — deterministic, no LLM needed
echo "[1a] Content modifier MRs..."
python3 scripts/01_dataset_construction/generate_safemt_modifier_followups.py

# 1b. NLP MRs (51/8/204/155) via llmorph
#     llmorph reads run_config.json from its own directory, so we cd into it.
#     With only_generate_followup_inputs=true it just writes the JSON files
#     without calling any LLM (inputs already exist as source_inputs_safemt.json).
echo "[1b] NLP MRs via llmorph..."
if [[ -d "llmorph/src" ]]; then
    pushd llmorph/src > /dev/null
    python3 mt_main.py
    popd > /dev/null
else
    echo "  [SKIP] llmorph not found — NLP MR follow-up files must already exist"
    echo "         in dataset/followup_prompts/"
fi

# =============================================================================
# STEP 2 — Image generation
# =============================================================================
if [[ "$SKIP_GENERATION" == true ]]; then
    log "STEP 2 — Image generation [SKIPPED]"
else
    log "STEP 2 — Image generation"

    for m in "${MODELS[@]}"; do
        echo
        echo "  ── Model: $m ──"
        case "$m" in
            sdxl)
                python3 scripts/02_image_generation/run_generation_batch.py \
                    --provider sdxl \
                    --prompt-set both \
                    --swarm-url "$SWARM_URL"
                ;;
            gpt)
                check_file "$OPENAI_KEY_FILE"
                OPENAI_API_KEY="$(cat "$OPENAI_KEY_FILE")" \
                python3 scripts/02_image_generation/run_generation_batch.py \
                    --provider openai \
                    --model gpt-image-1 \
                    --prompt-set both \
                    --quality medium
                ;;
            gemini)
                if [[ -z "$GEMINI_PROJECT" ]]; then
                    echo "  ERROR: --gemini-project required for gemini model"; exit 1
                fi
                python3 scripts/02_image_generation/run_generation_batch.py \
                    --provider gemini-vertex \
                    --model gemini-2.5-flash-image \
                    --prompt-set both \
                    --project-id "$GEMINI_PROJECT"
                ;;
        esac
    done
fi

# =============================================================================
# STEP 3 — Output relation check (3-layer consistency)
# =============================================================================
log "STEP 3 — Output relation check"

# Manifest paths (generated in step 2, or pre-existing in dataset/manifests/)
SDXL_MANIFESTS=(
    "data/generated/safemt/output/manifest_0_1350.jsonl"
    "data/generated/safemt/output_modifier/manifest_0_1080.jsonl"
)
GPT_MANIFEST="data/generated/safemt/openai_gpt-image-1/manifest.jsonl"
GEMINI_MANIFEST="data/generated/safemt/gemini-vertex_gemini-2_5-flash-image/manifest.jsonl"

for m in "${MODELS[@]}"; do
    echo
    echo "  ── Consistency check: $m ──"
    case "$m" in
        sdxl)
            python3 scripts/03_consistency_checking/output_relation.py \
                --manifest "${SDXL_MANIFESTS[@]}" \
                --output data/generated/safemt/sdxl_output_relation_results.jsonl
            ;;
        gpt)
            python3 scripts/03_consistency_checking/output_relation.py \
                --manifest "$GPT_MANIFEST" \
                --output data/generated/safemt/gpt_image_output_relation_results.jsonl
            ;;
        gemini)
            python3 scripts/03_consistency_checking/output_relation.py \
                --manifest "$GEMINI_MANIFEST" \
                --output data/generated/safemt/gemini_output_relation_results.jsonl
            ;;
    esac
done

# =============================================================================
# STEP 4 — Final report
# =============================================================================
log "STEP 4 — Final report"

python3 scripts/safemt_report.py --model "$MODEL" --output safemt_report.txt

echo
echo "Pipeline complete. Report saved to: safemt_report.txt"

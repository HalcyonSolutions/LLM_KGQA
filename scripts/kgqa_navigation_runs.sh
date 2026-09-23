#!/usr/bin/env bash

set -euo pipefail

# ============================================================
# Model profiles
# ============================================================

model_configs=(
    "configs/models/qwen3.json"
    "configs/models/gemma4.json"
    "configs/models/qwen2.5-instruct-q4.json"
    "configs/models/llama3.1-instruct-q4.json"
    "configs/models/granite3.3.json"
    "configs/models/ministral-3-instruct-q4.json"
    "configs/models/olmo-3-instruct.json"
    "configs/models/phi4-mini.json"
)

# ============================================================
# Datasets
# ============================================================

datasets=(
    "kinship"
    "mquake_single"
    "mquake_multi"
)

declare -A dataset_hops
declare -A dataset_max_steps
declare -A dataset_max_actions
declare -A dataset_context_window

dataset_hops["kinship"]="n"
dataset_max_steps["kinship"]=3
dataset_max_actions["kinship"]=100
dataset_context_window["kinship"]=$((8 * 1024))

dataset_hops["mquake_single"]="n"
dataset_max_steps["mquake_single"]=4
dataset_max_actions["mquake_single"]=200
dataset_context_window["mquake_single"]=$((32 * 1024))

dataset_hops["mquake_multi"]="n"
dataset_max_steps["mquake_multi"]=4
dataset_max_actions["mquake_multi"]=200
dataset_context_window["mquake_multi"]=$((32 * 1024))

dataset_hops["metaqa"]="n"
dataset_max_steps["metaqa"]=3
dataset_max_actions["metaqa"]=200
dataset_context_window["metaqa"]=$((32 * 1024))

# ============================================================
# Common experiment settings
# ============================================================

temperature=0
seed=42
timeout=15

# Use 0 for the structured-response benchmark so one logical
# decision always corresponds to one LLM generation.
max_parse_retries=0

run_navigation() {
    local dataset="$1"
    local model_config="$2"
    local prompting="$3"
    local structured="$4"

    local prompt_flags=()
    local output_flags=()

    case "$prompting" in
        zero-shot)
            prompt_flags+=(--prompting-approach zero-shot --n-shots 0)
            ;;
        one-shot)
            prompt_flags+=(
                --prompting-approach one-shot
                --n-shots 1
                --demo-history-mode full
                --demo-max-actions 5
            )
            ;;
        *)
            echo "Unknown prompting mode: $prompting"
            return 1
            ;;
    esac

    if [[ "$structured" == "true" ]]; then
        output_flags+=(--structured-output)
    fi

    echo
    echo "============================================================"
    echo "Dataset:          $dataset"
    echo "Model profile:    $model_config"
    echo "Prompting:        $prompting"
    echo "Structured:       $structured"
    echo "Max steps:        ${dataset_max_steps[$dataset]}"
    echo "Max actions:      ${dataset_max_actions[$dataset]}"
    echo "Context window:   ${dataset_context_window[$dataset]}"
    echo "Temperature:      $temperature"
    echo "Seed:             $seed"
    echo "============================================================"
    echo

    python ./kgqa_navigation.py \
        --dataset "$dataset" \
        --hops "${dataset_hops[$dataset]}" \
        --max-navigation-steps "${dataset_max_steps[$dataset]}" \
        --max-actions "${dataset_max_actions[$dataset]}" \
        --context-window "${dataset_context_window[$dataset]}" \
        --model-config "$model_config" \
        --navigation-approach tuple \
        --memory-approach full \
        "${prompt_flags[@]}" \
        "${output_flags[@]}" \
        --temperature "$temperature" \
        --seed "$seed" \
        --timeout "$timeout" \
        --timeout-cooldown 0 \
        --max-parse-retries "$max_parse_retries"
}

echo "Running KGQA Navigation Experiments"

echo
echo "### Zero-shot / Full Memory / Tuple / Structured ###"

for dataset in "${datasets[@]}"; do
    for model_config in "${model_configs[@]}"; do
        run_navigation "$dataset" "$model_config" "zero-shot" "true"
    done
done

echo
echo "### One-shot / Full Demo / Full Memory / Tuple / Structured ###"

for dataset in "${datasets[@]}"; do
    for model_config in "${model_configs[@]}"; do
        run_navigation "$dataset" "$model_config" "one-shot" "true"
    done
done

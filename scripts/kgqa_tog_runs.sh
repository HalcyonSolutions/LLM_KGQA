#!/usr/bin/env bash

set -euo pipefail

# ============================================================
# Model profiles
# ============================================================

# Top-performing SLMs selected from the local-navigation
# MQuAKE-ST Single results.
model_configs=(
    "configs/models/ministral-3-instruct-q4.json"
    "configs/models/qwen3.json"
    "configs/models/gemma4.json"
)

# ============================================================
# ToG-style evaluation settings
# ============================================================

dataset="mquake_single"
hops="n"
split="test"
max_depth=4
neighborhood_threshold=20
num_retain_entity=5
context_window=$((32 * 1024))
max_output_tokens=1024
temperature=0
seed=42
timeout=60
connect_timeout=5
max_parse_retries=0
checkpoint_every=10
result_dir="./results/tog"

widths=(1 3)

run_tog() {
    local model_config="$1"
    local width="$2"

    echo
    echo "============================================================"
    echo "ToG-style KGQA Evaluation"
    echo "Dataset:                $dataset"
    echo "Model profile:          $model_config"
    echo "Width:                  $width"
    echo "Max depth:              $max_depth"
    echo "Graph directionality:   outgoing"
    echo "Neighborhood threshold: $neighborhood_threshold"
    echo "Retained neighbors:     $num_retain_entity"
    echo "Context window:         $context_window"
    echo "Max output tokens:      $max_output_tokens"
    echo "Temperature:            $temperature"
    echo "Seed:                   $seed"
    echo "============================================================"
    echo

    python ./kgqa_tog.py \
        --dataset "$dataset" \
        --hops "$hops" \
        --split "$split" \
        --model-config "$model_config" \
        --width "$width" \
        --max-depth "$max_depth" \
        --neighborhood-threshold "$neighborhood_threshold" \
        --num-retain-entity "$num_retain_entity" \
        --context-window "$context_window" \
        --max-output-tokens "$max_output_tokens" \
        --temperature "$temperature" \
        --seed "$seed" \
        --timeout "$timeout" \
        --connect-timeout "$connect_timeout" \
        --max-parse-retries "$max_parse_retries" \
        --checkpoint-every "$checkpoint_every" \
        --result-dir "$result_dir"
}

echo "Running ToG-style MQuAKE-ST Single Experiments"

for width in "${widths[@]}"; do
    echo
    echo "### ToG-style Search / w=$width ###"
    for model_config in "${model_configs[@]}"; do
        run_tog "$model_config" "$width"
    done
done

echo
echo "============================================================"
echo "All ToG-style experiments completed."
echo "============================================================"

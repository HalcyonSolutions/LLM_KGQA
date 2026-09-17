#!/usr/bin/env bash

set -euo pipefail

# ============================================================
# Models
# ============================================================

# Top-performing SLMs selected from the local-navigation
# MQuAKE-ST Single results.
models=(
    # "ministral-3"
    "qwen3"
    "gemma4"
)

# Match the model artifacts used in the main navigation study.
requires_instruct=(
    "ministral-3"
)

requires_quantized=(
    "ministral-3"
)

# ============================================================
# ToG-style evaluation settings
# ============================================================

dataset="mquake_single"
hops="n"
split="test"

# MQuAKE-ST has reasoning paths up to four hops.
max_depth=4

# Original ToG-style neighborhood pruning settings used by
# the local adapter.
neighborhood_threshold=20
num_retain_entity=5

context_window=$((32 * 1024))
max_output_tokens=1024

temperature=0
seed=42

timeout=60
connect_timeout=5

# Do not introduce repair/re-prompt calls. A logical ToG
# decision should use only the original generation.
max_parse_retries=0

checkpoint_every=10
result_dir="./results/tog"

# Evaluate the single-retained-path condition first.
# w=3 is the default ToG beam width and is substantially
# more expensive.
widths=(
    1
    3
)

# ============================================================
# Helpers
# ============================================================

contains_model() {
    local target="$1"
    shift

    local item
    for item in "$@"; do
        if [[ "$item" == "$target" ]]; then
            return 0
        fi
    done

    return 1
}


run_tog() {
    local model="$1"
    local width="$2"

    local model_flags=()

    # --------------------
    # Model variant
    # --------------------

    if contains_model "$model" "${requires_instruct[@]}"; then
        model_flags+=(--use-instruct)
    fi

    if contains_model "$model" "${requires_quantized[@]}"; then
        model_flags+=(
            --use-quantized
            --quantization-bits 4
        )
    fi

    echo
    echo "============================================================"
    echo "ToG-style KGQA Evaluation"
    echo "Dataset:                $dataset"
    echo "Model:                  $model"
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
        --llm-model "$model" \
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
        --result-dir "$result_dir" \
        "${model_flags[@]}"
}


# ============================================================
# Experiments
# ============================================================

echo "Running ToG-style MQuAKE-ST Single Experiments"

# ------------------------------------------------------------
# 1. Width = 1
#
# Single retained search path. This is the closest ToG-style
# condition to the single-trajectory local-navigation setting.
# ------------------------------------------------------------

echo
echo "### ToG-style Search / w=1 ###"

for model in "${models[@]}"; do
    run_tog "$model" 1
done


# ------------------------------------------------------------
# 2. Width = 3
#
# Default ToG beam width. Run only after every w=1 experiment
# has completed because this condition is substantially more
# expensive.
# ------------------------------------------------------------

echo
echo "### ToG-style Search / w=3 ###"

for model in "${models[@]}"; do
    run_tog "$model" 3
done


echo
echo "============================================================"
echo "All ToG-style experiments completed."
echo "============================================================"
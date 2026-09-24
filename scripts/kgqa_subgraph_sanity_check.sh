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
    # "kinship"
    "mquake_single"
    "mquake_multi"
)

echo "Evaluation with evidence paths only"
for dataset in "${datasets[@]}"; do
    for model_config in "${model_configs[@]}"; do
        
        echo
        echo "============================================================"
        echo "Dataset:          $dataset"
        echo "Model profile:    $model_config"
        echo "============================================================"
        echo

        python ./kgqa_subgraph.py --dataset "$dataset" --hops n --model-config "$model_config" -e
    done
done

# Evaluation on KGQA Iterative Navigation on a small subset of questions

# Evaluation on MQuAKE (zero-shot, original navigation)
python ./kgqa_navigation.py \
    --model-config configs/models/qwen3.json \
    --dataset mquake_single \
    --hops n \
    --max-navigation-steps 4 \
    --prompting-approach zero-shot \
    --max-actions 200 \
    --context-window 32768 \
    --navigation-approach tuple \
    --memory-approach full \
    --result-dir ./results/navigation \
    --seed 42 \
    --temperature 0 \
    --timeout-cooldown 0 \
    --max-parse-retries 0 \
    --structured-output \
    --timeout 15 \
    --max-questions 10

# Evaluation on MQuAKE (one-shot, original navigation)
python ./kgqa_navigation.py \
    --model-config configs/models/qwen3.json \
    --dataset mquake_single \
    --hops n \
    --max-navigation-steps 4 \
    --prompting-approach one-shot \
    --n-shots 1 \
    --demo-history-mode full \
    --demo-max-actions 5 \
    --max-actions 200 \
    --context-window 32768 \
    --navigation-approach tuple \
    --memory-approach full \
    --result-dir ./results/navigation \
    --seed 42 \
    --temperature 0 \
    --timeout-cooldown 0 \
    --max-parse-retries 0 \
    --structured-output \
    --timeout 15 \
    --max-questions 10

# Evaluation using MQuAKE (ToG-style navigation)
python kgqa_tog.py \
    --model-config configs/models/qwen3.json \
    --dataset mquake_single \
    --hops n \
    --width 3 \
    --max-depth 4 \
    --neighborhood-threshold 20 \
    --num-retain-entity 5 \
    --context-window 32768 \
    --max-output-tokens 256 \
    --result-dir ./results/tog \
    --checkpoint-every 10 \
    --seed 42 \
    --temperature 0 \
    --connect-timeout 5 \
    --max-parse-retries 0 \
    --timeout 15 \
    --max-questions 10
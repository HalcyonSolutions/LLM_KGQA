# Evaluation on KGQA Iterative Navigation on a small subset of questions

# Evaluation on MQuAKE (zero-shot, original navigation)
python ./kgqa_navigation.py \
    --dataset mquake_single \
    --hops n \
    --max-navigation-steps 4 \
    --max-actions 200 \
    --context-window 32768 \
    --llm-model qwen2.5 \
    --use-instruct \
    --navigation-approach tuple \
    --memory-approach full \
    --prompting-approach zero-shot \
    --timeout 15

# Evaluation on MQuAKE (one-shot, original navigation)
python ./kgqa_navigation.py \
    --dataset mquake_single \
    --hops n \
    --max-navigation-steps 4 \
    --n-shots 1 \
    --demo-history-mode full \
    --demo-max-actions 5 \
    --max-actions 200 \
    --context-window 32768 \
    --llm-model qwen2.5 \
    --use-instruct \
    --navigation-approach tuple \
    --memory-approach full \
    --prompting-approach one-shot \
    --timeout 15

# Evaluation on MQuAKE (zero-shot, hybrid navigation)
python ./kgqa_navigation.py \
    --dataset mquake_single \
    --hops n \
    --max-navigation-steps 4 \
    --max-actions 200 \
    --context-window 32768 \
    --llm-model qwen2.5 \
    --use-instruct \
    --navigation-approach hybrid \
    --memory-approach full \
    --prompting-approach zero-shot \
    --timeout 15

# Evaluation on Kinship (zero-shot, original navigation)
python ./kgqa_navigation.py \
    --dataset kinship \
    --hops n \
    --max-navigation-steps 3 \
    --llm-model qwen2.5 \
    --use-instruct \
    --navigation-approach tuple \
    --memory-approach full \
    --prompting-approach zero-shot \
    --timeout 15

python ./kgqa_navigation.py \
    --dataset mquake_multi \
    --hops n \
    --max-navigation-steps 4 \
    --n-shots 1 \
    --demo-history-mode full \
    --demo-max-actions 5 \
    --max-actions 200 \
    --context-window 32768 \
    --llm-model gemma4 \
    --structured-output \
    --navigation-approach tuple \
    --memory-approach full \
    --prompting-approach one-shot \
    --timeout 15 \
    --max-questions 10 

python ./kgqa_navigation.py \
    --dataset mquake_single \
    --hops n \
    --max-navigation-steps 4 \
    --n-shots 1 \
    --demo-history-mode full \
    --demo-max-actions 5 \
    --max-actions 200 \
    --context-window 32768 \
    --llm-model deepseek-r1 \
    --navigation-approach tuple \
    --memory-approach full \
    --prompting-approach one-shot \
    --timeout 15

python kgqa_tog.py \
    --dataset mquake_single \
    --hops n \
    --split test \
    --llm-model ministral-3 \
    --width 3 \
    --max-depth 4 \
    --bidirectional \
    --neighborhood-threshold 20 \
    --num-retain-entity 5 \
    --context-window 32768 \
    --max-output-tokens 256 \
    --temperature 0 \
    --seed 42 \
    --timeout 15 \
    --connect-timeout 5 \
    --max-parse-retries 0 \
    --checkpoint-every 10 \
    --result-dir ./results/tog \
    --max-questions 10
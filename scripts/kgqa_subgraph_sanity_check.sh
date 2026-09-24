# Evaluation on Kinship with evidence paths only
python ./kgqa_subgraph.py --dataset kinship --hops n --model-config configs/models/llama3.json -e
python ./kgqa_subgraph.py --dataset kinship --hops n --model-config configs/models/llama3.1.json -e
python ./kgqa_subgraph.py --dataset kinship --hops n --model-config configs/models/deepseek-coder.json -e
python ./kgqa_subgraph.py --dataset kinship --hops n --model-config configs/models/gpt-oss.json -e
python ./kgqa_subgraph.py --dataset kinship --hops n --model-config configs/models/mixtral.json -e

# Evaluation on MQuAKE with evidence paths only
python ./kgqa_subgraph.py --dataset mquake_single --hops n --model-config configs/models/llama3.json -e
python ./kgqa_subgraph.py --dataset mquake_single --hops n --model-config configs/models/llama3.1.json -e
python ./kgqa_subgraph.py --dataset mquake_single --hops n --model-config configs/models/deepseek-coder.json -e
python ./kgqa_subgraph.py --dataset mquake_single --hops n --model-config configs/models/qwen2.5.json -e
python ./kgqa_subgraph.py --dataset mquake_single --hops n --model-config configs/models/gpt-oss.json -e
python ./kgqa_subgraph.py --dataset mquake_single --hops n --model-config configs/models/mixtral.json -e

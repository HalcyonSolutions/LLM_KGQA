# Subgraph QA

The subgraph QA workflow provides a sampled graph context to the language model and asks it to produce the final answer directly, rather than selecting graph actions hop by hop.

The main entry point is `kgqa_subgraph.py`. These utilities are retained from the broader KGQA codebase and are separate from the paper's primary navigation setting. Model identity is supplied with `--model-config`, with optional `--model-id` override for server-local aliases.

## Basic Usage

```bash
python ./kgqa_subgraph.py \
  --dataset mquake_single \
  --hops n \
  --model-config configs/models/qwen2.5-instruct-q4.json \
  --sampling-method neighborhood \
  --subgraph-size 50 \
  --max-depth 3 \
  --result-dir ./results
```

## Sampling Methods

- `neighborhood`: expand around evidence or source nodes.
- `random`: sample graph triplets while preserving required evidence seeds.
- `evidence`: use only evidence paths.

Use `-r` / `--retrieve` to use non-oracle retrieval from the source node.

## Outputs

Subgraph runs save the run configuration, aggregate statistics, and title-mapping status in the result artifact.

For the expected dataset layout, see [Datasets](datasets.md). For result organization and analysis utilities, see [Results and Analysis](results.md).

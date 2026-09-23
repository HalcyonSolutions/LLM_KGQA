# Reproducing the Paper Experiments

The shell scripts under `scripts/` encode the model, dataset, prompting, navigation, and inference settings used for the reported paper experiments. Run all commands from the repository root.

## Tables 2 and 3: Navigation

The one-shot navigation results and the zero-shot versus one-shot comparison are reproduced with:

```bash
bash scripts/kgqa_navigation_runs.sh
```

The script evaluates:

- datasets: Kinship, MQuAKE-ST Single, and MQuAKE-ST Multi;
- models: Qwen3, Gemma4, Qwen2.5, Llama 3.1, Granite 3.3, Ministral-3, OLMo-3, and Phi-4 Mini;
- tuple navigation with full memory;
- structured responses;
- both zero-shot and one-shot prompting;
- seed 42 and temperature 0.

Dataset-specific limits encoded by the script are:

| Dataset | Max navigation steps | Max actions | Context window |
| --- | ---: | ---: | ---: |
| Kinship | 3 | 100 | 8K |
| MQuAKE-ST Single | 4 | 200 | 32K |
| MQuAKE-ST Multi | 4 | 200 | 32K |

The script also records model-specific instruct and quantized variants used in the experiments.

## Table 4: Think-on-Graph

The ToG-style comparison is reproduced with:

```bash
bash scripts/kgqa_tog_runs.sh
```

The script evaluates MQuAKE-ST Single using the selected top-performing local-navigation models:

- Ministral-3;
- Qwen3;
- Gemma4.

It evaluates widths `1` and `3`, with maximum depth `4`. Width 1 is run first because width 3 is substantially more expensive.

The script fixes the paper settings for neighborhood pruning, output limits, timeouts, seed, and model variants. It also disables repair/re-prompt calls so one logical ToG decision corresponds to one generation.

## Additional Experiment Scripts

Sample navigation runs:

```bash
bash scripts/kgqa_navigation_samples.sh
```

Subgraph sanity checks:

```bash
bash scripts/kgqa_subgraph_sanity_check.sh
```

A repository reproducibility check is also kept under `scripts/`.

## Backend Configuration

Before running experiments, copy the backend template:

```bash
cp configs/openwebui_template.json configs/openwebui_config.json
```

Then fill in the connection information for Ollama or Open WebUI.

For workflow-specific options, see [Navigation](navigation.md), [Think-on-Graph](tog.md), and [Subgraph QA](subgraph.md).

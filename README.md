# LLMs KGQA Project

## Overview

This repository is the **official implementation** for the paper **“The Path Matters: Evaluating Small Language Models Beyond Answer Accuracy in KGQA”** by Eduin E. Hernandez, Sergio A. Diaz, Luis F. Garcia, Nurassyl Askar, and Stefano Rini. The preprint is available on [arXiv:2609.27669](https://arxiv.org/abs/2609.27669).

The paper evaluates frozen, locally deployable small language models (SLMs) as question-conditioned graph-navigation policies, measuring terminal-answer accuracy together with executed-path fidelity. It also includes a Think-on-Graph (ToG)-style comparison to study the effect of explicit search and answer-generation scaffolding.

The repository supports three KGQA workflows:

- **Iterative navigation QA**: the LLM selects legal graph actions step by step. This is the paper's primary experimental setting.
- **Think-on-Graph (ToG)**: a local adaptation of the original ToG search procedure used for the paper's secondary search comparison.
- **Subgraph QA**: the LLM receives a sampled subgraph and predicts the final answer directly.

The main entry points are `kgqa_navigation.py`, `kgqa_tog.py`, and `kgqa_subgraph.py`.

## Installation

Python 3.12 or newer is required.

```bash
pip install -r requirements.txt
```

### Backend Setup

The experiments can use either **Ollama directly** or **Open WebUI backed by Ollama**.

#### 1. Install Ollama and download a model

Install Ollama from [ollama.com/download](https://ollama.com/download). Ollama normally exposes its local API at `http://localhost:11434`.

Download the exact model variant referenced by a model profile. For example:

```bash
ollama pull qwen3:8b
```

Confirm that it is installed:

```bash
ollama list
```

The model ID reported by Ollama should match the profile's `model.id` whenever possible. If your server uses a different alias, the runners also accept `--model-id`.

#### 2. Choose a backend

A configuration template is provided at:

```text
configs/openwebui_template.json
```

Copy it before editing:

```bash
cp configs/openwebui_template.json configs/openwebui_config.json
```

`configs/openwebui_config.json` is local configuration and should not be committed.

##### Direct Ollama

For the simplest local setup, call Ollama directly:

```json
{
  "backend": "ollama",
  "ollama_url": "http://localhost:11434"
}
```

No API key is required in direct Ollama mode.

##### Open WebUI

[Open WebUI](https://github.com/open-webui/open-webui) provides a browser interface and an authenticated API in front of Ollama. Follow the [official installation guide](https://docs.openwebui.com/getting-started/quick-start/) to install it. The Docker quick start exposes Open WebUI at `http://localhost:3000`; a Python installation commonly uses `http://localhost:8080`.

After Open WebUI is running:

1. Open **Settings > Admin > Connections**.
2. Under **Manage Ollama API Connections**, connect the Ollama instance. If Open WebUI runs directly on the same host, this is usually `http://localhost:11434`. If Open WebUI runs in Docker while Ollama runs on the host, use the host address recommended by the Open WebUI documentation (commonly `http://host.docker.internal:11434`).
3. Open **Manage** on the Ollama connection to verify that the downloaded model is visible. Models can also be downloaded from this panel or directly from the model selector.
4. If the connection uses a **Model IDs** filter, either leave it empty to expose all models from that Ollama instance or add the exact model ID used by the profile.
5. Under **Settings > Admin > Models**, make sure the model is enabled and visible to the users who will run the experiments. Adjust public/group access if the Open WebUI instance is shared.

For example, after pulling `qwen3:8b`, that exact model ID should be available through Open WebUI before running a profile whose `model.id` is `qwen3:8b`.

#### 3. Enable and create an Open WebUI API key

API keys are disabled globally unless an administrator enables them:

1. Log in as an Open WebUI administrator.
2. Go to **Settings > Admin > Authentication**.
3. Enable **API Keys** and save the settings.
4. For non-admin accounts, also grant the **API Keys** feature permission through the appropriate user/group permissions.
5. Open your profile menu and go to **Settings > Account**.
6. In the **API keys** section, click **Show**, then **Create new secret key**.
7. Copy the generated key and keep it private.

Do not commit an Open WebUI API key to the repository.

Configure the repository to use Open WebUI:

```json
{
  "backend": "openwebui",
  "base_url": "http://localhost:8080",
  "api_key": "sk-<OPENWEBUI_API_KEY>"
}
```

Set `base_url` to the URL at which your Open WebUI instance is actually reachable (for example, `http://localhost:3000` for the standard Docker quick start).

You can then verify model access through the repository by running one of the sample or sanity-check commands below.

### Model Profiles

Models are configured through validated JSON profiles under `configs/models/` rather than repository-specific model-name conventions.

A profile records the backend model ID, model family, instruct/quantization metadata, context-window limit, and declared capabilities. Select one with:

```bash
--model-config configs/models/qwen2.5-instruct-q4.json
```

If another server exposes the same deployed model under a different ID, override only the API-facing identifier:

```bash
--model-config configs/models/qwen2.5-instruct-q4.json \
--model-id my-server-model-alias
```

See [configs/models/README.md](configs/models/README.md) for the profile schema and validation rules.

## Datasets

The paper evaluates the navigation-ready **KINSHIP** and **MQuAKE-ST** resources, including the Single Answer and Multi Answer MQuAKE-ST settings.

Dataset releases, preparation details, and associated THESEUS resources are maintained in the [THESEUS repository](https://github.com/HalcyonSolutions/THESEUS). For paper reproduction, use the dataset versions distributed through THESEUS.

Local datasets are expected under `data/<dataset_name>/`. The `data/` directory is not part of the checked-in repository tree.

See [docs/datasets.md](docs/datasets.md) for the expected file layout and optional entity/relation mappings.

## Reproducing the Paper Experiments

Run the experiment scripts from the repository root.

### Tables 2 and 3: Navigation

The one-shot navigation results and the zero-shot versus one-shot comparison are reproduced with:

```bash
bash scripts/kgqa_navigation_runs.sh
```

### Table 4: Think-on-Graph

The ToG-style search comparison is reproduced with:

```bash
bash scripts/kgqa_tog_runs.sh
```

These scripts contain the model, dataset, prompting, navigation/search, and inference settings used for the reported experiments.

See [docs/reproducibility.md](docs/reproducibility.md) for the exact configurations encoded by the scripts.

## Quick Usage

### Iterative Navigation QA

```bash
python ./kgqa_navigation.py \
  --dataset mquake_single \
  --hops n \
  --model-config configs/models/qwen2.5-instruct-q4.json \
  --navigation-approach tuple \
  --memory-approach full \
  --prompting-approach zero-shot \
  --n-shots 0 \
  --max-navigation-steps 4 \
  --max-actions 200 \
  --result-dir ./results
```

Navigation supports tuple, factorized, and hybrid action selection together with zero-shot and n-shot prompting.

See [docs/navigation.md](docs/navigation.md) for prompting, memory, action-selection, truncation, context-window, and debugging options.

### Think-on-Graph

```bash
python ./kgqa_tog.py \
  --dataset mquake_single \
  --model-config configs/models/qwen3.json
```

The local adapter preserves the upstream ToG prompt family while adapting graph access, entity/relation mappings, evaluation, and result logging to this repository.

See [docs/tog.md](docs/tog.md) for the upstream revision, search procedure, fallback behavior, parsing policy, answer evaluation, implementation differences, audit trail, and result versioning.

### Subgraph QA

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

See [docs/subgraph.md](docs/subgraph.md) for sampling and retrieval options.

## Repository Structure

```text
analysis/   result compilation, diagnostics, comparisons, and plotting
configs/    API/backend configuration templates
docs/       detailed workflow and reproducibility documentation
model/      shared and task-specific LLM clients
scripts/    experiment, sample, sanity-check, and reproducibility scripts
tests/      unit, integration, and behavior tests
tools/      standalone debugging and development utilities
utils/      graph, metrics, KGQA, parsing, and API utilities
```

The local `data/` directory and generated `results/` directory are not checked into the repository.

For a module-level overview, see [docs/development.md](docs/development.md).

## Results and Analysis

Generated outputs are written under `results/`.

Navigation results include the run configuration, aggregate statistics, per-question episodes, executed paths, path-fidelity and final-entity metrics, and prompt/action truncation metadata. ToG results additionally retain search, answer-source, parsing, and formatting audit information.

Common analysis commands include:

```bash
python analysis/compile_navigation_results.py --dataset kinship
python analysis/plot_metric.py --dataset mquake
```

See [docs/results.md](docs/results.md) for result fields, ToG audit metadata, and the available analysis utilities.

## Documentation

Detailed documentation is available under [docs/](docs/README.md):

- [Navigation](docs/navigation.md)
- [Think-on-Graph](docs/tog.md)
- [Subgraph QA](docs/subgraph.md)
- [Datasets](docs/datasets.md)
- [Reproducibility](docs/reproducibility.md)
- [Results and Analysis](docs/results.md)
- [Development](docs/development.md)

## Citation

If you use this repository, please cite:

```bibtex
@article{hernandez2026path,
  title         = {The Path Matters: Evaluating Small Language Models Beyond Answer Accuracy in KGQA},
  author        = {Eduin E. Hernandez and Sergio A. Diaz and Luis F. Garcia and Nurassyl Askar and Stefano Rini},
  year          = {2026},
  journal       = {arXiv preprint arXiv:2609.27669}
}
```

## License

This project is licensed under the [Apache License 2.0](LICENSE).

Third-party datasets, model weights, and other external resources remain
subject to their respective licenses and terms of use.
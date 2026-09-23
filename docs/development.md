# Development Guide

## Repository Structure

```text
analysis/   result compilation, diagnostics, comparisons, and plotting
configs/    backend configuration templates
model/      shared and task-specific LLM clients
scripts/    experiment, sample, sanity-check, and reproducibility scripts
tests/      unit, integration, and behavior tests
tools/      standalone debugging and development utilities
utils/      graph, metrics, KGQA, parsing, and API utilities
```

The local `data/` and generated `results/` directories are not part of the checked-in repository tree.

## Main Entry Points

- `kgqa_navigation.py`: iterative graph-navigation KGQA.
- `kgqa_tog.py`: local Think-on-Graph baseline.
- `kgqa_subgraph.py`: subgraph-at-once KGQA.

## Model Clients

- `model/base_llm_client.py`: shared LLM client functionality.
- `model/navigation_llm_client.py`: navigation prompts, parsing, and control logic.
- `model/subgraph_llm_client.py`: subgraph prompt and prediction logic.
- `model/tog_llm_client.py`: ToG-specific model interaction and selection logic.
- `model/tog_original_prompts.py`: preserved upstream ToG prompt templates.

## Shared Utilities

- `utils/kgqa_utils.py`: shared KGQA helpers, including optional title-map loading.
- `utils/kgqa_types.py`: common KGQA type aliases.
- `utils/kgqa_navigation_metrics.py`: navigation answer and path metrics.
- `utils/kgqa_navigation_utils.py`: navigation-specific utilities.
- `utils/tog_search.py`: local ToG graph-search implementation.
- `utils/tog_parsing.py`: ToG response parsing and formatting-tolerance logic.
- `utils/action_selection.py`: action-selection utilities.
- `utils/graph_utils.py`: graph utilities.
- `utils/api_utils.py`: API helpers.

## Tests

The `tests/` directory currently includes coverage for multi-answer navigation, ToG search behavior, ToG parsing, grounded ToG behavior, run-file checks, and remote/API behavior.

Run the relevant tests from the repository root using your preferred Python test runner or direct test invocation as appropriate.

## Development Utilities

The `tools/` directory contains standalone helpers used for debugging and experimentation:

- `ollama_playground.py`;
- `preview_action_selection.py`;
- `qa_subgraph_sampling.py`.

These utilities are intentionally separated from automated tests.

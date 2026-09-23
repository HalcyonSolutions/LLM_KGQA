# LLMs KGQA Project

## Overview

This repository is the **official implementation** for the paper **“The Path Matters: Evaluating Small Language Models Beyond Answer Accuracy in KGQA”** by Eduin E. Hernandez, Sergio A. Diaz, Luis F. Garcia, Nurassyl Askar, and Stefano Rini.

The paper evaluates frozen, locally deployable small language models (SLMs) as question-conditioned graph-navigation policies, measuring terminal-answer accuracy together with executed-path fidelity. It also includes a secondary Think-on-Graph (ToG)-style comparison to study the effect of explicit search and answer-generation scaffolding.

The repository contains the following KGQA workflows:

- **Iterative navigation QA**: let the LLM choose graph actions step by step while the controller validates and executes only legal KG edges. This is the primary experimental setting used in the paper.
- **Think-on-Graph (ToG)**: adapt the original ToG search procedure to the local KGQA datasets and graph representation for the paper's secondary search comparison.
- **Subgraph QA**: provide the LLM with a sampled subgraph and ask for the final answer. These utilities are retained from the broader KGQA codebase.

The main runners are `kgqa_navigation.py`, `kgqa_tog.py`, and `kgqa_subgraph.py`.

## Project Structure

- `analysis/`: result compilation, diagnostics, comparisons, and plotting utilities.
- `configs/`: API/backend configuration files.
- `data/`: local KGQA dataset files.
- `model/`: shared base client and task-specific LLM clients.
- `results/`: JSON outputs and derived experiment artifacts.
- `scripts/`: Bash scripts for experiment runs, samples, sanity checks, and reproducibility checks.
- `tests/`: unit, integration, and behavior tests.
- `tools/`: standalone Python debugging and development utilities.
- `utils/`: reusable graph, metric, KGQA, parsing, and API utilities.

## Key Files

- `kgqa_navigation.py`: iterative graph-navigation KGQA runner used for the paper's primary experiments.
- `kgqa_tog.py`: local Think-on-Graph baseline runner used for the paper's secondary search comparison.
- `kgqa_subgraph.py`: subgraph-at-once KGQA runner.
- `model/base_llm_client.py`: shared LLM client logic.
- `model/navigation_llm_client.py`: navigation-specific prompt, parsing, and control logic.
- `model/subgraph_llm_client.py`: subgraph-specific prompt and prediction logic.
- `model/tog_llm_client.py`: ToG-specific LLM interaction and selection logic.
- `utils/kgqa_utils.py`: shared KGQA helpers, including optional title-map loading.
- `utils/kgqa_types.py`: shared KGQA type aliases.
- `utils/kgqa_navigation_metrics.py`: navigation answer/path metrics.
- `utils/tog_search.py`: local ToG graph-search implementation.
- `utils/tog_parsing.py`: ToG response parsing and formatting-tolerance utilities.

## Installation

This project requires Python 3.12 or higher.

```bash
pip install -r requirements.txt
```

### API Configuration

A configuration template is provided at:

```text
configs/openwebui_template.json
```

Before running the experiments, make a copy of the template named:

```text
configs/openwebui_config.json
```

Then edit `configs/openwebui_config.json` and fill in the missing connection information for your selected backend.

The configuration supports either direct Ollama access or Open WebUI through the backend field. For Open WebUI, provide the server address and API key. For Ollama, provide the Ollama server URL.

## Data Layout

Each dataset should live under `data/<dataset_name>/` and include:

- `triplets.txt`
- `qa_<hops>hop.csv`, for example `qa_nhop.csv`

Encoded datasets such as MQuAKE may also include:

- `node_data.csv`
- `relation_data.csv`

These mapping files are optional. If they are missing, as in unencoded datasets such as `kinship`, the runners assume entity and relation strings are already readable and omit title mappings from prompts.

### Datasets Used in the Paper

The paper evaluates the navigation-ready **KINSHIP** and **MQuAKE-ST** resources, including the Single Answer and Multi Answer MQuAKE-ST settings. The dataset releases, preparation details, and associated THESEUS resources are maintained in the [THESEUS repository](https://github.com/HalcyonSolutions/THESEUS).

For reproducing the paper experiments, use the dataset versions distributed through THESEUS rather than reconstructing them from the generic layout description above.

## Usage

### Iterative Navigation QA

```bash
python ./kgqa_navigation.py \
  --dataset mquake_single \
  --hops n \
  --llm-model qwen2.5 \
  --use-instruct \
  --navigation-approach tuple \
  --memory-approach full \
  --prompting-approach zero-shot \
  --n-shots 0 \
  --demo-history-mode full \
  --demo-max-actions 10 \
  --max-navigation-steps 4 \
  --max-actions 200 \
  --result-dir ./results
```

Navigation modes:

- `tuple`: the LLM chooses directly from full outgoing edge actions.
- `factorized`: the LLM chooses a relation first, then receives the normal action prompt limited to only that relation's edges.
- `hybrid`: uses tuple mode for small neighborhoods and the same two-stage factorized flow for larger ones.

`--n-shots` prepends complete solved train trajectories to action-selection prompts. Demonstration sampling is seed-reproducible and prefers longer train trajectories so later hops show gold path history. One shot is one complete trajectory: the question and start entity are shown once, then each hop shows the current entity, selected history view, available actions, and gold JSON selected action. `--n-shots 0` preserves the existing zero-shot behavior. `--prompting-approach one-shot` is a convenience alias for `--n-shots 1` when no explicit shot count is provided.

`--demo-history-mode` controls demonstration history independently of test-time navigation memory. Use `full` for all previous gold hops, `last` for only the immediately previous hop, or `random` for one seeded random previous hop. Hops with no previous edge show `(none)`.

`--demo-max-actions` caps the number of legal actions shown in each demonstrated hop. The demonstrated action list always contains the gold next edge, with other options filled from the remaining sorted neighborhood, and the gold `{"action": ...}` ID is recomputed after truncation. This cap is independent from `--max-actions` and does not change test-time navigation logic.

`--max-actions` caps the options shown in each inference prompt. Use `--max-actions-policy first` (default), `random` (seeded sampling), or `question-aware` (deterministic lexical ranking). Results record prompt-local and original sorted option IDs.

The prompt is still checked against `--context-window`; if the prompt, including demonstrations, is too large, the episode terminates with `context_window_exceeded` before that LLM call.

Use `--show-navigation` or `--show-actions` to print each prompt, model response, validated move, and termination reason.

### Subgraph QA

```bash
python ./kgqa_subgraph.py \
  --dataset mquake_single \
  --hops n \
  --llm-model qwen2.5 \
  --use-instruct \
  --sampling-method neighborhood \
  --subgraph-size 50 \
  --max-depth 3 \
  --result-dir ./results
```

Subgraph sampling modes:

- `neighborhood`: expand around evidence or source nodes.
- `random`: sample graph triplets while preserving required evidence seeds.
- `evidence`: use only evidence paths.

Use `-r` / `--retrieve` for non-oracle retrieval from the source node.

### Experiment and Utility Scripts

Commands are intended to be run from the repository root.

Experiment shell scripts are under `scripts/`, for example:

```bash
bash scripts/kgqa_navigation_runs.sh
bash scripts/kgqa_tog_runs.sh
bash scripts/kgqa_navigation_samples.sh
```

Result-analysis utilities are under `analysis/`, for example:

```bash
python analysis/compile_navigation_results.py --dataset kinship
python analysis/plot_metric.py --dataset mquake
```

Development and debugging utilities are under `tools/`, while automated and integration tests are kept under `tests/`.

## Results

Results are saved under `results/<dataset>/`.

Navigation outputs include:

- run configuration
- aggregate statistics
- per-question episode records
- selected actions and readable executed paths
- path-fidelity and final-entity metrics
- title-mapping status
- truncation/context-window metadata

Subgraph outputs include:

- run configuration
- aggregate statistics
- title-mapping status

## Notes

- The graph controller only executes legal KG edges listed in the prompt.
- Navigation uses the terminal graph entity as the prediction.
- `first` and `question-aware` are deterministic; `random` is reproducible for the same seed, question, step, stage, and current entity.
- No answer-type hints are added to prompts.

## Think-on-Graph baseline

`kgqa_tog.py` adapts the original ToG Freebase algorithm to our local triples,
entity/relation title maps, dataset rows, and LLM API client. Its reference is
[ToG revision 7ccbb92](https://github.com/DataArcTech/ToG/tree/7ccbb92e17579f934bb778386230de47eca0ab67/ToG).
The upstream prompt module is preserved verbatim in `model/tog_original_prompts.py`
with its source revision. We use its Freebase relation, entity, reasoning,
partial-evidence answer, and question-only CoT templates.

The default flow is bidirectional relation pruning, per-relation entity scoring,
global beam pruning by the current relation score times entity score, and
reasoning over all retained triples across depths. The model's successful
reasoning response is reused as the final answer. Neighborhoods of at least 20
entities are randomly reduced to 5. Defaults are width 3, depth 3, exploration
temperature 0.4, reasoning temperature 0, and 256 output tokens, matching the
reference script. Entity-score count mismatches use uniform scores; score totals
are not enforced. Relation parse failures produce no selected relations.

Fallback behavior also follows upstream:

- Successful sufficiency check: return that reasoning response and its answer.
- No further candidates: answer using accumulated triples **and model knowledge**.
- Depth exhausted without a successful stop: answer from the original CoT prompt
  and question **without retrieved triples**.
- No starting entities: use the same question-only fallback.

Answers are free text and may refer to entities outside the local graph. Exact,
unambiguous label/ID resolution against the full local title map is an optional
postprocessing step; it never constrains generation and never uses gold answers.
The original response is retained as `answer_reasoning`; its braced answer is
extracted into `generated_answer`. `answer_source` and aggregate `answer_sources`
distinguish reasoning, partial-evidence fallback, and knowledge-only fallback.
Search traces remain stored even when the fallback does not consume them.

Our framework reports **generated-answer Hits@1**: the extracted response matches
the normalized gold answer text (including supplied text aliases), or its
unambiguously resolved entity ID belongs to the gold entity set. Unresolved text
can still be correct. Errors and unmatched responses count as zero. This is
stricter than upstream's evaluation helper, which also accepts substring matches;
we do not label the scores as an exact reproduction of the paper's evaluation.
Terminal-entity accuracy, final-frontier beam hit rate, and top search-path
fidelity remain separate diagnostics. The beam width is not an answer count.

Remaining framework adaptations: local graph snapshot instead of SPARQL;
one dataset `Source-Entity` in the runner; relation labels decorated with local
IDs to disambiguate names; raw ID fallback for missing labels instead of
Freebase unnamed-entity/metadata filters; deterministic neighborhood sampling;
explicit association of previous direction with each unique frontier entity;
and bounded API retries via our model backend. The supplied LLM and graph/data
also differ from the original experiments. Search does not use answer annotations
unless `--oracle-selectors` is explicitly enabled.

Example:

```bash
python kgqa_tog.py --dataset mquake_single --llm-model qwen3 --use-instruct
```

Use `--max-depth 4` for four-hop experiments. `--temperature` overrides both stage
temperatures; `--no-bidirectional` and `--disable-early-stop` are ablations.
`--structured-output` is rejected because it conflicts with the original text
prompts. The legacy `--max-parse-retries` flag now controls failed API/text-response
retries only, not semantic score retries. A no-API smoke check is:

```bash
python kgqa_tog.py --oracle-selectors --max-depth 4 --max-questions 5 \
  --result-dir /tmp/tog-smoke
```

New outputs use `results_v4_...`, `method_version=tog_original_local_v4`, and a
pinned `upstream_commit`. Old unversioned terminal-entity scores and v2 restricted
entity-answer scores must be rerun. `analysis/compare_navigation_tog.py` rejects
mixing ToG method versions or answer metrics across beam-width conditions.

### Formatting tolerance and audit trail

The original prompts and fallback policy are unchanged. `utils/tog_parsing.py`
adds a versioned, gold-independent formatting adapter. Scored relation selections
can use braces, Markdown, parenthesized/bracketed IDs, or an unambiguous exact
relation label. IDs must belong to the available options. Unknown IDs, ambiguous
labels, duplicate choices, and invalid scores are rejected rather than guessed.

Answers can be braced, bare, in an explicit final-answer clause, or a final
asserted bold value. Arbitrary mentions elsewhere in the explanation are not
answers; ambiguity/refusal is not resolved by looking at the gold text. A plain
or Markdown leading Yes/No is accepted for sufficiency. Entity IDs can be resolved
from an exact label, ID, or a consistent label-plus-ID pair in the full graph map.
Correctness is still evaluated after extraction with the existing exact matcher;
we do not use upstream-style unrestricted substring credit.

Calls retain their raw output and `parsing`/`answer_parsing` metadata. Episodes
include `formatting_events`, `answer_parsing`, and `entity_resolution`; aggregate
`formatting_counts` reports strict, tolerant, rejected, and fallback events.
Counts describe parser events, not necessarily failed questions. Uniform scoring
fallbacks are explicitly logged. Relation events include the selected ID,
direction, score, and original line. Answer events record the extraction method
and resulting answer. `parser_version` records the adapter version.

`python analysis/audit_tog_formatting.py INPUT --output OUTPUT` replays answer extraction into a
separate diagnostic artifact, retaining old and new predictions. It does not
rewrite the run, rerun search, or estimate accuracy of navigation that would have
changed under the new parser. The saved Ministral audit is in
`results/analysis/ministral_v3_formatting_audit.json`. Rerun navigation to assess
relation-formatting fixes. Old v3 results must not be mixed with v4 runs.

## License

This project is licensed under the Academic License.

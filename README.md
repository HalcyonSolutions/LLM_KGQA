# LLMs KGQA Project

## Overview

This project runs Large Language Model (LLM) experiments for Knowledge Graph Question Answering (KGQA). It supports two task styles:

- **Subgraph QA**: provide the LLM with a sampled subgraph and ask for the final answer.
- **Iterative navigation QA**: let the LLM choose graph actions step by step while the controller validates and executes only legal KG edges.

The current runners are `kgqa_subgraph.py` and `kgqa_navigation.py`.

## Project Structure

- `configs/`: API/backend configuration files.
- `data/`: KGQA datasets.
- `model/`: shared base client and task-specific LLM clients.
- `results/`: JSON outputs from experiment runs.
- `utils/`: shared typing, graph, metric, KGQA, and API utilities.
- `scripts/`: helper scripts for experiments and sanity checks.

## Key Files

- `kgqa_navigation.py`: iterative graph-navigation KGQA runner.
- `kgqa_subgraph.py`: subgraph-at-once KGQA runner.
- `model/base_llm_client.py`: shared LLM client logic.
- `model/navigation_llm_client.py`: navigation-specific prompt, parsing, and control logic.
- `model/subgraph_llm_client.py`: subgraph-specific prompt and prediction logic.
- `utils/kgqa_utils.py`: shared KGQA helpers, including optional title-map loading.
- `utils/kgqa_types.py`: shared KGQA type aliases.
- `utils/kgqa_navigation_metrics.py`: navigation answer/path metrics.

## Installation

This project requires Python 3.12 or higher.

```bash
pip install -r requirements.txt
```

The scripts expect an API configuration in `configs/openwebui_config.json`.

## Data Layout

Each dataset should live under `data/<dataset_name>/` and include:

- `triplets.txt`
- `qa_<hops>hop.csv`, for example `qa_nhop.csv`

Encoded datasets such as MQuAKE may also include:

- `node_data.csv`
- `relation_data.csv`

These mapping files are optional. If they are missing, as in unencoded datasets such as `kinship_v2`, the runners assume entity and relation strings are already readable and omit title mappings from prompts.

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

## License

This project is licensed under the Academic License.

## TODO

### Navigation

- [ ] Implement `--prompting-approach io` and keep the prompt-template interface extensible for future prompt families.
- [ ] Add graph directionality options (`outgoing`, `incoming`, `bidirectional`) and propagate the setting through action indexing, path validation, metrics, and result config.
- [x] Add configurable `--max-actions` selection policies, such as `first`, seeded random sampling, and question-aware ranking.
- [ ] Record both prompt-local option IDs and original sorted graph-action IDs in episode records, especially for truncated tuple prompts and factorized relation-action prompts.
- [ ] Reverify n-shot demonstrations for factorized and hybrid navigation flows.
- [ ] Add support for `--memory-approach none` in n-shot demonstration prompts.
- [ ] Add adaptive context-window handling before termination, such as reducing shown actions or switching from tuple to factorized prompts when possible.
- [ ] Build a human navigation GUI for manually stepping through the graph and answering questions.
- [x] Reuse the standard action prompt for factorized second-stage navigation over the selected relation's edges.
- [x] Add n-shot navigation demonstrations from complete train-set gold trajectories.

### Prompting And LLM Calls

- [ ] Add shared one-shot and few-shot prompt templates for subgraph QA.
- [ ] Add zero-context QA evaluation without graph context.
- [ ] Add an optional final-answer generation call after navigation instead of always using the terminal entity as the answer.
- [ ] Add an optional triplet-to-sentence prompt format for subgraph evidence.
- [ ] Add bulk/batch chat execution where supported by the backend.
- [ ] Verify LLM cancellation and model unload behavior across normal completion, timeout, and interruption.
- [x] Set LLM seed through the backend.
- [x] Add timeout and retry handling for long LLM calls.
- [x] Add prompt token estimation and context-window configuration.
- [x] Add support for multiple LLM backends, including OpenWebUI and Ollama.
- [ ] Add an option to select thinking mode for LLM calls, which may improve response quality at the cost of more tokens and slower responses. Currently hardcoded into api.

### Metrics And Evaluation

- [ ] Revisit path-fidelity metrics and document the intended PED, RED, F1_SG, and F1_REL behavior.
- [x] Ensure PED and F1_SG correctly support multiple gold answers and multiple valid evidence paths.
- [x] Support multiple-answer and multiple-valid-path navigation evaluation end to end.
- [ ] Add validation-split evaluation for navigation hyperparameter tuning.
- [ ] Revisit final-answer cleanup punctuation rules so abbreviations such as `U.S.A.` are not corrupted.
- [ ] Add regression tests for navigation termination reasons, parse retries, max-action truncation, and context-window failures.
- [ ] Double-check deterministic subgraph prompts across models when seeds are fixed.

### Data And Architecture

- [ ] Add dataset compatibility checks for required columns and optional fields before launching long runs.
- [ ] Allow configurable title-map column names, such as `QID`/`Property` or `EID`/`RID`, for datasets like MetaQA.
- [x] Decide whether navigation parsing should accept JSON responses with `action` and `stop` plus extra fields, or continue requiring the exact schema.
- [ ] Add support notes or adapters for additional KGQA datasets such as MetaQA and PathQuestion.
- [x] Support datasets with optional entity/relation title mappings.
- [x] Split shared LLM client logic from subgraph and navigation task clients.

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
entity-answer scores must be rerun. `compare_navigation_tog.py` rejects mixing
ToG method versions or answer metrics across beam-width conditions.

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

`audit_tog_formatting.py INPUT --output OUTPUT` replays answer extraction into a
separate diagnostic artifact, retaining old and new predictions. It does not
rewrite the run, rerun search, or estimate accuracy of navigation that would have
changed under the new parser. The saved Ministral audit is in
`results/analysis/ministral_v3_formatting_audit.json`. Rerun navigation to assess
relation-formatting fixes. Old v3 results must not be mixed with v4 runs.

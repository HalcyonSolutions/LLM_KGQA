## TODO

### Navigation

- [ ] Implement `--prompting-approach io` and keep the prompt-template interface extensible for future prompt families.
- [ ] Add graph directionality options (`outgoing`, `incoming`, `bidirectional`) and propagate the setting through action indexing, path validation, metrics, and result config.
- [x] Add configurable `--max-actions` selection policies, such as `first`, seeded random sampling, and question-aware ranking.
- [x] Record both prompt-local option IDs and original sorted graph-action IDs in episode records, especially for truncated tuple prompts and factorized relation-action prompts.
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
- [x] Add an option to select thinking mode for LLM calls, which may improve response quality at the cost of more tokens and slower responses. Currently hardcoded into api.

### Metrics And Evaluation

- [ ] Revisit path-fidelity metrics and document the intended PED, RED, F1_SG, and F1_REL behavior.
- [x] Ensure PED and F1_SG correctly support multiple gold answers and multiple valid evidence paths.
- [x] Support multiple-answer and multiple-valid-path navigation evaluation end to end.
- [ ] Add validation-split evaluation for navigation hyperparameter tuning.
- [ ] Revisit final-answer cleanup punctuation rules so abbreviations such as `U.S.A.` are not corrupted.
- [ ] Add regression tests for navigation termination reasons, parse retries, max-action truncation, and context-window failures.
- [ ] Double-check deterministic subgraph prompts across models when seeds are fixed.

### Data And Architecture

- [x] Add a dataset preprocessing/download script that fetches the supported datasets from Hugging Face (or a configured mirror) and materializes the required files under `./data/<dataset_name>/` using the repository's expected dataset structure.
- [x] Replace hard-coded per-model constants and parameter definitions with validated model profile files referenced at runtime, including sanity checks for required fields, context limits, capabilities, and backend model resolution.
- [ ] Add dataset compatibility checks for required columns and optional fields before launching long runs.
- [ ] Allow configurable title-map column names, such as `QID`/`Property` or `EID`/`RID`, for datasets like MetaQA.
- [x] Decide whether navigation parsing should accept JSON responses with `action` and `stop` plus extra fields, or continue requiring the exact schema.
- [ ] Add support notes or adapters for additional KGQA datasets such as MetaQA and PathQuestion.
- [x] Support datasets with optional entity/relation title mappings.
- [x] Split shared LLM client logic from subgraph and navigation task clients.

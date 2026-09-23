# Think-on-Graph Baseline

`kgqa_tog.py` adapts the original Think-on-Graph (ToG) Freebase algorithm to the repository's local triples, entity/relation title maps, dataset rows, and LLM API client.

The implementation references [ToG revision 7ccbb92](https://github.com/DataArcTech/ToG/tree/7ccbb92e17579f934bb778386230de47eca0ab67/ToG). The upstream prompt module is preserved verbatim in `model/tog_original_prompts.py` together with its source revision. The local adapter uses the original Freebase relation, entity, reasoning, partial-evidence answer, and question-only CoT templates.

## Search Procedure

The default flow performs:

1. bidirectional relation pruning;
2. per-relation entity scoring;
3. global beam pruning using the current relation score multiplied by the entity score;
4. reasoning over all retained triples accumulated across search depths.

When the model determines that the retained evidence is sufficient, that successful reasoning response is reused as the final answer.

Neighborhoods containing at least 20 entities are randomly reduced to 5. The reference defaults are width 3, depth 3, exploration temperature 0.4, reasoning temperature 0, and 256 output tokens.

Entity-score count mismatches fall back to uniform scores. Score totals are not otherwise enforced. Relation parse failures produce no selected relations.

## Fallback Behavior

The local implementation follows the upstream fallback policy:

- Successful sufficiency check: return that reasoning response and its answer.
- No further candidates: answer using accumulated triples **and model knowledge**.
- Depth exhausted without a successful stop: answer from the original question-only CoT prompt **without retrieved triples**.
- No starting entities: use the same question-only fallback.

Answers are free text and may refer to entities outside the local graph.

Exact, unambiguous label/ID resolution against the complete local title map is available as optional postprocessing. Resolution never constrains generation and never uses gold answers.

The original response is retained as `answer_reasoning`; its extracted answer is stored in `generated_answer`. `answer_source` and aggregate `answer_sources` distinguish reasoning, partial-evidence fallback, and knowledge-only fallback. Search traces are retained even when the final fallback does not consume them.

## Answer Evaluation

The framework reports **generated-answer Hits@1**.

A generated answer is counted as correct when either:

- the extracted response matches normalized gold answer text, including supplied text aliases; or
- its unambiguously resolved entity ID belongs to the gold entity set.

Unresolved text can therefore still be correct. Errors and unmatched responses count as zero.

This is stricter than the upstream evaluation helper, which also accepts substring matches. These scores should therefore not be described as an exact reproduction of the original ToG paper evaluation.

Terminal-entity accuracy, final-frontier beam hit rate, and top search-path fidelity are kept as separate diagnostics. Beam width is not treated as an answer count.

## Local Framework Adaptations

The local implementation differs from the original environment in several necessary ways:

- a local graph snapshot is used instead of SPARQL;
- the runner uses one dataset `Source-Entity`;
- relation labels are decorated with local IDs to disambiguate names;
- missing labels fall back to raw IDs rather than Freebase unnamed-entity/metadata filters;
- neighborhood sampling is deterministic;
- previous direction is explicitly associated with each unique frontier entity;
- API retries are bounded by the shared model backend;
- the supplied language models and graph/data differ from the original experiments.

Search does not use answer annotations unless `--oracle-selectors` is explicitly enabled.

## Usage

Basic example:

```bash
python kgqa_tog.py --dataset mquake_single --llm-model qwen3 --use-instruct
```

Use `--max-depth 4` for four-hop experiments.

`--temperature` overrides both stage temperatures. `--no-bidirectional` and `--disable-early-stop` are available as ablations.

`--structured-output` is rejected because it conflicts with the original text prompts.

The legacy `--max-parse-retries` flag controls failed API/text-response retries only, not semantic score retries.

A no-API smoke check is available with oracle selectors:

```bash
python kgqa_tog.py --oracle-selectors --max-depth 4 --max-questions 5 \
  --result-dir /tmp/tog-smoke
```

## Prompt Compatibility

The local adapter intentionally retains the original ToG Freebase-style prompt family. The upstream wiki-oriented prompt family is not interchangeable with this adapter because the local implementation operates on its human-readable relation-label and local-ID interface.

## Formatting Tolerance

The original prompts and fallback policy are unchanged. `utils/tog_parsing.py` adds a versioned, gold-independent formatting adapter around their outputs.

Scored relation selections may use braces, Markdown, parenthesized/bracketed IDs, or an unambiguous exact relation label. IDs must belong to the available options.

Unknown IDs, ambiguous labels, duplicate choices, and invalid scores are rejected rather than guessed.

Answers may be extracted from:

- braced output;
- bare output;
- an explicit final-answer clause;
- a final asserted bold value.

Arbitrary mentions elsewhere in the explanation are not treated as answers, and ambiguity/refusal is not resolved using the gold answer.

A plain or Markdown leading Yes/No is accepted for sufficiency checks.

Entity IDs may be resolved from an exact label, an ID, or a consistent label-plus-ID pair in the full graph map. Correctness is still evaluated only after extraction using the repository's exact matcher; unrestricted upstream-style substring credit is not used.

## Audit Trail

LLM calls retain raw output together with `parsing` or `answer_parsing` metadata.

Episodes include:

- `formatting_events`;
- `answer_parsing`;
- `entity_resolution`.

Aggregate `formatting_counts` records strict, tolerant, rejected, and fallback events. These counts describe parser events and do not necessarily correspond one-to-one with failed questions.

Uniform-scoring fallbacks are logged explicitly. Relation events retain the selected ID, direction, score, and original line. Answer events record the extraction method and extracted answer. `parser_version` identifies the formatting adapter version.

Historical answer extraction can be replayed with:

```bash
python analysis/audit_tog_formatting.py INPUT --output OUTPUT
```

The audit writes a separate diagnostic artifact containing old and new predictions. It does not rewrite the run, rerun search, or estimate the navigation accuracy that would have changed under a different relation parser.

A saved Ministral v3 audit is located at:

```text
results/analysis/ministral_v3_formatting_audit.json
```

## Result Versioning

Current outputs use:

```text
results_v4_...
method_version=tog_original_local_v4
```

They also record the pinned upstream commit.

Old unversioned terminal-entity scores and v2 restricted entity-answer scores must be rerun. Old v3 results must not be mixed with v4 runs.

`analysis/compare_navigation_tog.py` rejects attempts to combine incompatible ToG method versions or answer metrics across beam-width conditions.

For the exact Table 4 experiment settings, see [Reproducibility](reproducibility.md).

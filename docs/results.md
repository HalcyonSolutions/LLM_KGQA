# Results and Analysis

Generated experiment outputs are written under `results/`. This directory is generated locally and is not part of the checked-in repository tree.

## Navigation Outputs

Navigation result artifacts include:

- run configuration;
- aggregate statistics;
- per-question episode records;
- selected actions and readable executed paths;
- path-fidelity and final-entity metrics;
- title-mapping status;
- action-truncation metadata;
- context-window metadata.

## Subgraph Outputs

Subgraph result artifacts include:

- run configuration;
- aggregate statistics;
- title-mapping status.

## ToG Outputs

ToG results retain the generated answer together with search and parsing metadata. Important fields include:

- `answer_reasoning`: the original final model response used for answer extraction;
- `generated_answer`: the extracted answer string;
- `answer_source`: whether the answer came from reasoning, partial-evidence fallback, or knowledge-only fallback;
- `answer_sources`: aggregate counts by answer source;
- search traces, even when the final fallback does not consume them;
- formatting and parser metadata described in [Think-on-Graph](tog.md).

## Analysis Utilities

Commands are intended to be run from the repository root.

Compile navigation results:

```bash
python analysis/compile_navigation_results.py --dataset kinship
```

Plot a metric:

```bash
python analysis/plot_metric.py --dataset mquake
```

Compare navigation and ToG result sets with:

```bash
python analysis/compare_navigation_tog.py ...
```

Audit historical ToG answer formatting with:

```bash
python analysis/audit_tog_formatting.py INPUT --output OUTPUT
```

The ToG audit utility writes a separate diagnostic artifact and does not rewrite the original run, rerun graph search, or estimate navigation accuracy that would have changed under a newer relation parser.

Additional result diagnostics are available in `analysis/check_results_errors.py`.

For compatibility rules between ToG result versions, see [Think-on-Graph](tog.md).

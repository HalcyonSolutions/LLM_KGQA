# Iterative Navigation QA

The iterative navigation workflow treats the language model as a question-conditioned graph-action policy. At each hop, the environment exposes legal outgoing actions from the current entity, the model selects an action, and the controller validates and executes only edges that exist in the local knowledge graph.

The main entry point is `kgqa_navigation.py`. Model identity and deployment metadata are supplied through a validated JSON profile under `configs/models/`; `--model-id` can override only the server-specific API identifier.

## Basic Usage

```bash
python ./kgqa_navigation.py \
  --dataset mquake_single \
  --hops n \
  --model-config configs/models/qwen3.json \
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

## Navigation Approaches

- `tuple`: the model selects directly from complete outgoing edge actions.
- `factorized`: the model selects a relation first, then receives the standard action prompt restricted to edges using that relation.
- `hybrid`: tuple mode is used for smaller neighborhoods, while larger neighborhoods use the same two-stage factorized flow.

## Demonstration Prompting

`--n-shots` prepends complete solved training trajectories to the action-selection prompts. Demonstration sampling is seed-reproducible and prefers longer training trajectories so that later hops contain meaningful gold path history.

One shot corresponds to one complete trajectory. The question and start entity are shown once, after which every hop includes the current entity, the selected history view, the available actions, and the gold JSON action.

`--n-shots 0` preserves zero-shot behavior. `--prompting-approach one-shot` is a convenience alias for `--n-shots 1` when no explicit shot count is supplied.

### Demonstration History

`--demo-history-mode` controls demonstration history independently from test-time navigation memory:

- `full`: show all previous gold hops.
- `last`: show only the immediately previous hop.
- `random`: show one seeded random previous hop.

Hops without a previous edge show `(none)`.

### Demonstration Action Cap

`--demo-max-actions` limits the legal actions displayed in each demonstrated hop. The demonstrated action list always retains the gold next edge; the remaining slots are filled from the sorted neighborhood. The gold `{"action": ...}` ID is recomputed after truncation.

This setting is independent of `--max-actions` and does not modify test-time navigation behavior.

## Test-Time Action Limits

`--max-actions` limits the options shown in each inference prompt. Supported policies include:

- `first` (default): retain the first actions from the deterministic sorted neighborhood.
- `random`: seeded action sampling.
- `question-aware`: deterministic lexical ranking based on the question.

Results retain both prompt-local action IDs and the original sorted graph-action IDs so truncated prompts can be audited.

## Context Window Handling

Prompts are checked against `--context-window` before each model call. If the complete prompt, including demonstrations, exceeds the configured context window, the episode terminates with `context_window_exceeded` before the LLM is called.

## Debugging

Use `--show-navigation` or `--show-actions` to print prompts, model responses, validated actions, and termination reasons while an episode is running.

## Evaluation Notes

- The graph controller executes only legal KG edges listed in the prompt.
- The terminal graph entity is used as the navigation prediction.
- No answer-type hints are inserted into the prompts.
- `first` and `question-aware` are deterministic.
- `random` is reproducible for the same seed, question, step, stage, and current entity.

For the exact paper experiment settings, see [Reproducibility](reproducibility.md).

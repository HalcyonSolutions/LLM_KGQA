"""Compile paired direct-navigation versus ToG width-1/width-3 results."""

import argparse
import csv
import json
import math
from pathlib import Path


PATH_METRICS = (
    "PED",
    "RED",
    "F1_SG",
    "F1_REL",
    "path_exact_match",
    "relation_chain_exact_match",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compile paired navigation/ToG result files for Table 3."
    )
    parser.add_argument(
        "--run",
        action="append",
        nargs=4,
        metavar=("MODEL", "DIRECT_JSON", "TOG_W1_JSON", "TOG_W3_JSON"),
        required=True,
        help="One model label and its three result files; repeat for multiple models.",
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Use at most the first N aligned episodes from each model.",
    )
    parser.add_argument(
        "--allow-partial-overlap",
        action="store_true",
        help="Use the question-ID intersection instead of failing on mismatched files.",
    )
    parser.add_argument("--output-json", default="table3_navigation_vs_tog.json")
    parser.add_argument("--output-csv", default="table3_navigation_vs_tog.csv")
    return parser.parse_args()


def load_result(path):
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or not isinstance(payload.get("episodes"), list):
        raise ValueError(f"{path} is not a result object containing an episodes list")
    return payload


def index_episodes(path, episodes):
    indexed = {}
    for position, episode in enumerate(episodes):
        question_id = episode.get("question_index", episode.get("question_number"))
        if question_id is None:
            raise ValueError(f"{path}: episode {position} has no question identifier")
        if question_id in indexed:
            raise ValueError(f"{path}: duplicate question identifier {question_id!r}")
        indexed[question_id] = episode
    return indexed


def mean(values):
    numeric = [float(value) for value in values if isinstance(value, (int, float))]
    return sum(numeric) / len(numeric) if numeric else None


def correctness(episode):
    return bool(episode.get("answer_correct", False))


def summarize_condition(episodes, condition):
    count = len(episodes)
    correct = sum(correctness(episode) for episode in episodes)
    failures = sum(
        episode.get("navigation_status") not in (None, "success")
        for episode in episodes
    )
    summary = {
        "condition": condition,
        "questions": count,
        "correct": correct,
        "accuracy": correct / count if count else 0.0,
        "failures": failures,
        "failure_rate": failures / count if count else 0.0,
        "llm_calls_total": sum(float(ep.get("actual_llm_calls", 0) or 0) for ep in episodes),
        "llm_calls_per_question": mean([ep.get("actual_llm_calls", 0) for ep in episodes]),
        "prompt_tokens_total": sum(float(ep.get("prompt_tokens", 0) or 0) for ep in episodes),
        "completion_tokens_total": sum(float(ep.get("completion_tokens", 0) or 0) for ep in episodes),
        "total_tokens": sum(float(ep.get("total_tokens", 0) or 0) for ep in episodes),
        "tokens_per_question": mean([ep.get("total_tokens", 0) for ep in episodes]),
        "wall_time_total": sum(float(ep.get("elapsed_time", 0) or 0) for ep in episodes),
        "wall_time_per_question": mean([ep.get("elapsed_time", 0) for ep in episodes]),
    }
    if condition.startswith("tog"):
        beam_correct = sum(bool(ep.get("beam_answer_hit", False)) for ep in episodes)
        generated_supported = [
            ep for ep in episodes if ep.get("generated_answer") is not None
        ]
        summary.update({
            "beam_correct": beam_correct,
            "beam_accuracy": beam_correct / count if count else 0.0,
            "generated_answer_support": len(generated_supported),
            "generated_answer_correct": sum(
                bool(ep.get("generated_answer_correct", False))
                for ep in generated_supported
            ),
        })
    for metric in PATH_METRICS:
        summary[f"path_{metric}"] = mean([
            (ep.get("path_fidelity") or {}).get(metric) for ep in episodes
        ])
    return summary


def exact_mcnemar_p(direct_only, tog_only):
    discordant = direct_only + tog_only
    if discordant == 0:
        return 1.0
    lower = min(direct_only, tog_only)
    tail = sum(math.comb(discordant, k) for k in range(lower + 1))
    return min(1.0, 2.0 * tail / (2 ** discordant))


def paired_summary(direct, tog, label):
    pairs = list(zip(direct, tog))
    both_correct = sum(correctness(a) and correctness(b) for a, b in pairs)
    direct_only = sum(correctness(a) and not correctness(b) for a, b in pairs)
    tog_only = sum(not correctness(a) and correctness(b) for a, b in pairs)
    both_wrong = len(pairs) - both_correct - direct_only - tog_only
    direct_accuracy = mean([correctness(ep) for ep in direct]) or 0.0
    tog_accuracy = mean([correctness(ep) for ep in tog]) or 0.0
    return {
        "comparison": label,
        "questions": len(pairs),
        "both_correct": both_correct,
        "direct_only_correct": direct_only,
        "tog_only_correct": tog_only,
        "both_wrong": both_wrong,
        "direct_accuracy": direct_accuracy,
        "tog_accuracy": tog_accuracy,
        "accuracy_delta_tog_minus_direct": tog_accuracy - direct_accuracy,
        "mcnemar_exact_p": exact_mcnemar_p(direct_only, tog_only),
        "llm_calls_delta_per_question": (
            (mean([ep.get("actual_llm_calls", 0) for ep in tog]) or 0.0)
            - (mean([ep.get("actual_llm_calls", 0) for ep in direct]) or 0.0)
        ),
        "tokens_delta_per_question": (
            (mean([ep.get("total_tokens", 0) for ep in tog]) or 0.0)
            - (mean([ep.get("total_tokens", 0) for ep in direct]) or 0.0)
        ),
        "wall_time_delta_per_question": (
            (mean([ep.get("elapsed_time", 0) for ep in tog]) or 0.0)
            - (mean([ep.get("elapsed_time", 0) for ep in direct]) or 0.0)
        ),
    }


def config_identity(payload):
    config = payload.get("config", {})
    return {
        "method_version": config.get("method_version"),
        "answer_metric_primary": config.get("answer_metric_primary"),
        "dataset": config.get("dataset"),
        "hop_split": config.get("hop_split", config.get("hops")),
        "model": config.get("model", config.get("llm_model")),
        "use_instruct": config.get("use_instruct"),
        "use_quantized": config.get("use_quantized"),
        "quantization_bits": config.get("quantization_bits"),
        "seed": config.get("seed"),
        "structured_output": config.get("structured_output"),
        "context_window": config.get("context_window"),
        "temperature": config.get("temperature"),
        "exploration_temperature": config.get("exploration_temperature"),
        "reasoning_temperature": config.get("reasoning_temperature"),
        "max_output_tokens": config.get("max_output_tokens"),
        "max_depth": config.get("max_navigation_steps", config.get("max_depth")),
        "graph_directionality": config.get("graph_directionality"),
        "max_parse_retries": config.get("max_parse_retries"),
        "neighborhood_threshold": config.get("neighborhood_threshold"),
        "num_retain_entity": config.get("num_retain_entity"),
        "disable_early_stop": config.get("disable_early_stop"),
    }


def validate_shared_config(model_label, identities):
    for field in identities["direct"]:
        # ToG's bidirectional search and reasoning output budget are method settings.
        if field in {"graph_directionality", "max_output_tokens", "method_version", "answer_metric_primary"}:
            continue
        observed = {
            identity[field]
            for identity in identities.values()
            if identity.get(field) is not None
        }
        if len(observed) > 1:
            raise ValueError(
                f"{model_label}: shared configuration mismatch for {field}: "
                f"{ {name: identity.get(field) for name, identity in identities.items()} }"
            )


def compile_model(model_label, paths, max_questions, allow_partial):
    names = ("direct", "tog_width1", "tog_width3")
    payloads = {name: load_result(path) for name, path in zip(names, paths)}
    indices = {
        name: index_episodes(path, payloads[name]["episodes"])
        for name, path in zip(names, paths)
    }
    id_sets = {name: set(index) for name, index in indices.items()}
    common = set.intersection(*id_sets.values())
    union = set.union(*id_sets.values())
    if common != union and not allow_partial:
        counts = {name: len(ids) for name, ids in id_sets.items()}
        raise ValueError(
            f"{model_label}: question IDs do not match: {counts}; "
            "use --allow-partial-overlap to analyze the intersection"
        )

    # Preserve direct-run order rather than assuming identifiers are sortable.
    ordered_ids = [
        question_id for question_id in indices["direct"] if question_id in common
    ]
    if max_questions is not None:
        ordered_ids = ordered_ids[:max_questions]
    aligned = {
        name: [indices[name][question_id] for question_id in ordered_ids]
        for name in names
    }
    for question_id in ordered_ids:
        question_texts = {
            str(indices[name][question_id].get("question", "")).strip()
            for name in names
        }
        if len(question_texts) != 1:
            raise ValueError(
                f"{model_label}: question text mismatch for ID {question_id!r}"
            )

    tog_metrics = {
        payloads[name].get("config", {}).get("answer_metric_primary", "top1_terminal_entity")
        for name in ("tog_width1", "tog_width3")
    }
    if len(tog_metrics) != 1:
        raise ValueError("Cannot mix different ToG answer metrics")
    tog_versions = {payloads[name].get("config", {}).get("method_version", "legacy")
                    for name in ("tog_width1", "tog_width3")}
    if len(tog_versions) != 1:
        raise ValueError("Cannot mix different ToG method versions")

    identities = {
        name: config_identity(payloads[name]) for name in names
    }
    validate_shared_config(model_label, identities)

    width1 = payloads["tog_width1"].get("config", {}).get("width")
    width3 = payloads["tog_width3"].get("config", {}).get("width")
    if width1 not in (None, 1):
        raise ValueError(f"{model_label}: ToG width-1 file reports width={width1}")
    if width3 not in (None, 3):
        raise ValueError(f"{model_label}: ToG width-3 file reports width={width3}")

    summaries = [
        summarize_condition(aligned[name], name) for name in names
    ]
    comparisons = [
        paired_summary(aligned["direct"], aligned["tog_width1"], "direct_vs_tog_width1"),
        paired_summary(aligned["direct"], aligned["tog_width3"], "direct_vs_tog_width3"),
        paired_summary(aligned["tog_width1"], aligned["tog_width3"], "tog_width1_vs_width3"),
    ]
    return {
        "model_label": model_label,
        "files": dict(zip(names, paths)),
        "configs": identities,
        "alignment": {
            "questions_used": len(ordered_ids),
            "common_questions_available": len(common),
            "union_questions": len(union),
            "missing_by_condition": {
                name: len(union - ids) for name, ids in id_sets.items()
            },
            "max_questions": max_questions,
        },
        "conditions": summaries,
        "paired_comparisons": comparisons,
    }


def write_csv(path, compiled):
    fields = [
        "model", "condition", "questions", "correct", "accuracy",
        "beam_correct", "beam_accuracy", "failures", "failure_rate",
        "llm_calls_total", "llm_calls_per_question", "total_tokens",
        "tokens_per_question", "wall_time_total", "wall_time_per_question",
        "path_PED", "path_RED", "path_F1_SG", "path_F1_REL",
        "path_path_exact_match", "path_relation_chain_exact_match",
    ]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for model in compiled:
            for condition in model["conditions"]:
                writer.writerow({"model": model["model_label"], **condition})


def main():
    args = parse_args()
    if args.max_questions is not None and args.max_questions < 1:
        raise ValueError("--max-questions must be positive")
    compiled = [
        compile_model(label, paths, args.max_questions, args.allow_partial_overlap)
        for label, *paths in args.run
    ]
    output = {
        "analysis": "direct_navigation_vs_tog_search",
        "max_questions": args.max_questions,
        "models": compiled,
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
    write_csv(args.output_csv, compiled)

    print("model\tcondition\tn\taccuracy\tbeam_accuracy\tcalls/q\ttokens/q\tseconds/q")
    for model in compiled:
        for row in model["conditions"]:
            print(
                f"{model['model_label']}\t{row['condition']}\t{row['questions']}\t"
                f"{row['accuracy']:.4f}\t"
                f"{'' if row.get('beam_accuracy') is None else f'{row['beam_accuracy']:.4f}'}\t"
                f"{row['llm_calls_per_question'] or 0:.2f}\t"
                f"{row['tokens_per_question'] or 0:.2f}\t"
                f"{row['wall_time_per_question'] or 0:.2f}"
            )
    print(f"JSON: {args.output_json}")
    print(f"CSV: {args.output_csv}")


if __name__ == "__main__":
    main()

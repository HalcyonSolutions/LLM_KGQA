"""Run Think-on-Graph beam search over an LLM_KGQA dataset snapshot."""

import argparse
import ast
import json
import os
import time
from pathlib import Path

from tqdm import tqdm

from model.constants import valid_models
from model.tog_llm_client import ToGLLMKGQAClient
from model.tog_original_prompts import UPSTREAM_COMMIT
from utils.tog_parsing import PARSER_VERSION
from utils.basic import load_pandas, load_triplets
from utils.graph_utils import Grapher
from utils.kgqa_data_utils import (
    get_row_value,
    normalize_answer_entities,
    normalize_reference_paths,
    normalize_relation_chain,
    to_jsonable,
)
from utils.kgqa_navigation_metrics import (
    aggregate_answer_metrics,
    aggregate_single_prediction_metrics,
    score_answer_set,
    score_single_final_entity,
)
from utils.kgqa_navigation_utils import best_path_fidelity_score, validate_executed_path
from utils.kgqa_utils import extract_final_answer, load_title_maps, translate_path
from utils.tog_search import LocalToGGraph, ToGSearchResult, run_tog_search


def parse_args():
    parser = argparse.ArgumentParser(description="Think-on-Graph over local KGQA graphs")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--dataset", default="mquake_single")
    parser.add_argument("--hops", default="n")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-questions", type=int)
    parser.add_argument("--width", type=int, default=3)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--bidirectional", action="store_true",
                        help="Search both directions instead of only outgoing edges. By default, our implementation uses only outgoing edges.")
    parser.add_argument("--structured-output", action="store_true")
    parser.add_argument("--neighborhood-threshold", type=int, default=20)
    parser.add_argument("--num-retain-entity", type=int, default=5)
    parser.add_argument("--disable-early-stop", action="store_true")
    parser.add_argument("--oracle-selectors", action="store_true", help="Use annotations for a no-API integration smoke test")
    parser.add_argument("--fail-fast", action="store_true", help="Stop instead of recording a failed question")
    parser.add_argument("--llm-model", choices=valid_models, default="gemma3")
    parser.add_argument("--use-instruct", action="store_true")
    parser.add_argument("--use-quantized", action="store_true")
    parser.add_argument("--quantization-bits", type=int, default=4)
    parser.add_argument("--context-window", type=int, default=4096)
    parser.add_argument("--max-output-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=None,
                        help="Override both stage temperatures (ablation)")
    parser.add_argument("--exploration-temperature", type=float, default=0.4)
    parser.add_argument("--reasoning-temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--connect-timeout", type=int, default=5)
    parser.add_argument("--max-parse-retries", type=int, default=2)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--result-dir", default="./results/tog")
    return parser.parse_args()


def sum_call_field(calls, field):
    return sum(float(call.get(field, 0) or 0) for call in calls)


def mean(values):
    return sum(values) / len(values) if values else 0.0


def summarize_calls_by_stage(calls):
    summary = {}
    for call in calls:
        stage = call.get("stage", "unknown")
        stage_summary = summary.setdefault(stage, {
            "actual_llm_calls": 0,
            "api_retries": 0,
            "prompt_tokens": 0.0,
            "completion_tokens": 0.0,
            "total_tokens": 0.0,
            "api_total_seconds": 0.0,
            "max_available_options": 0,
        })
        stage_summary["actual_llm_calls"] += 1
        stage_summary["api_retries"] += int(call.get("attempt", 0) > 0)
        stage_summary["prompt_tokens"] += float(call.get("prompt_tokens", 0) or 0)
        stage_summary["completion_tokens"] += float(call.get("response_tokens", 0) or 0)
        stage_summary["total_tokens"] += float(call.get("total_tokens", 0) or 0)
        stage_summary["api_total_seconds"] += float(call.get("total_seconds", 0) or 0)
        stage_summary["max_available_options"] = max(
            stage_summary["max_available_options"],
            int(call.get("available_options", 0) or 0),
        )
    return summary


def answer_text_correct(prediction, gold_answer):
    if prediction is None:
        return False
    predicted = extract_final_answer(str(prediction), lower=True)
    gold = gold_answer
    if isinstance(gold, str) and gold.strip().startswith("["):
        try:
            gold = ast.literal_eval(gold)
        except (SyntaxError, ValueError):
            pass
    gold = extract_final_answer(gold, lower=True)
    if isinstance(gold, list):
        return predicted in gold
    return predicted == gold


def reference_data(row, grapher, gold_entities):
    paths = normalize_reference_paths(get_row_value(row, "Paths"))
    relation_chain = normalize_relation_chain(get_row_value(row, "Path-Key"))
    source = "dataset_paths" if paths else None
    if not paths and relation_chain:
        paths = grapher.find_paths_by_relation_chain(
            start_entity=str(row["Source-Entity"]),
            relation_chain=relation_chain,
            target_entities=gold_entities,
        )
        if paths:
            source = "lazy_relation_chain"
    return paths, relation_chain, source


def write_payload(path, config, statistics, episodes):
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(to_jsonable({
            "config": config,
            "statistics": statistics,
            "episodes": episodes,
        }), handle, indent=2)
    os.replace(temporary, path)


def main():
    args = parse_args()
    if args.structured_output:
        raise ValueError("Original ToG uses free-text prompts; omit --structured-output")
    if args.temperature is not None:
        args.exploration_temperature = args.reasoning_temperature = args.temperature
    if args.width < 1 or args.max_depth < 0 or args.max_parse_retries < 0:
        raise ValueError("width must be positive; depth and parse retries must be non-negative")
    if args.neighborhood_threshold < 1 or args.num_retain_entity < 1:
        raise ValueError("neighborhood threshold and retained entity count must be positive")
    if args.max_questions is not None and args.max_questions < 1:
        raise ValueError("--max-questions must be positive")

    dataset_dir = os.path.join(args.data_dir, args.dataset)
    all_triplets = set(map(tuple, load_triplets(os.path.join(dataset_dir, "triplets.txt")).values))
    graph = LocalToGGraph(all_triplets)
    grapher = Grapher(all_triplets)
    entity_title, relation_title, mapping_status = load_title_maps(
        os.path.join(dataset_dir, "node_data.csv"),
        os.path.join(dataset_dir, "relation_data.csv"),
    )
    qa = load_pandas(os.path.join(dataset_dir, f"qa_{args.hops}hop.csv"))
    qa = qa[qa["SplitLabel"] == args.split]
    if args.max_questions is not None:
        qa = qa.head(args.max_questions)
    qa = qa.reset_index(drop=False).rename(columns={"index": "dataframe_index"})

    output_dir = os.path.join(args.result_dir, args.dataset)
    os.makedirs(output_dir, exist_ok=True)
    model_name = "oracle-smoke" if args.oracle_selectors else args.llm_model
    if not args.oracle_selectors and args.use_instruct:
        model_name += "-instruct"
        if args.use_quantized:
            model_name += f"-q{args.quantization_bits}"
    model_name = model_name.replace("/", "-").replace(":", "-")
    structured = "structured" if args.structured_output else "unstructured"
    direction = "bidirectional" if args.bidirectional else "outgoing"
    stopping = "noearlystop" if args.disable_early_stop else "earlystop"
    question_limit = f"_questions{len(qa)}" if args.max_questions is not None else ""
    output = os.path.join(
        output_dir,
        f"results_v4_{args.hops}hop_{model_name}_tog_w{args.width}_d{args.max_depth}_"
        f"nt{args.neighborhood_threshold}_nr{args.num_retain_entity}_"
        f"{direction}_{structured}_{stopping}_{args.split}"
        f"{question_limit}_seed{args.seed}.json",
    )
    config = vars(args) | {
        "title_mapping": mapping_status,
        "graph_directionality": "bidirectional" if args.bidirectional else "outgoing",
        "method_version": "tog_original_local_v4",
        "answer_metric_primary": "generated_answer_hits1",
        "answer_policy": "original_text_with_model_knowledge_fallback",
        "evidence_policy": "all_retained_triples_by_depth",
        "path_metric_scope": "top1_search_path_not_generated_answer",
        "upstream_reference": f"https://github.com/DataArcTech/ToG/tree/{UPSTREAM_COMMIT}/ToG",
        "upstream_commit": UPSTREAM_COMMIT,
        "parser_version": PARSER_VERSION,
        "answer_matcher": "exact_normalized_text_or_unambiguous_entity_id",
        "beam_metric": f"Hits@{args.width}",
    }

    client = None
    if not args.oracle_selectors:
        client = ToGLLMKGQAClient(
            Path(__file__).parent / "configs" / "openwebui_config.json",
            model_choice=args.llm_model,
            use_instruct=args.use_instruct,
            use_quantized=args.use_quantized,
            quantization_bits=args.quantization_bits,
            context_window=args.context_window,
            seed=args.seed,
            temperature=args.reasoning_temperature,
            exploration_temperature=args.exploration_temperature,
            reasoning_temperature=args.reasoning_temperature,
            timeout=args.timeout,
            connect_timeout=args.connect_timeout,
            max_output_tokens=args.max_output_tokens,
            entity_title=entity_title,
            relation_title=relation_title,
            max_parse_retries=args.max_parse_retries,
            structured_output=args.structured_output,
        )

    episodes = []
    generated_answer_scores = []
    beam_scores = []
    path_scores = []
    wall_times = []
    calls_per_question = []
    tokens_per_question = []
    failures = 0

    with tqdm(range(len(qa)), desc="ToG questions") as progress:
        for row_position in progress:
            row = qa.iloc[row_position]
            question = row["Question"]
            start = str(row["Source-Entity"])
            gold = normalize_answer_entities(row["Answer-Entity"])
            reference_paths, relation_chain, reference_source = reference_data(row, grapher, gold)
            call_start = len(client.calls) if client is not None else 0
            started = time.perf_counter()
            error = None
            result = None
            generated_answer = None
            predicted_answer_entity = None
            answer_reasoning = None
            answer_source = None
            answer_parsing = None
            entity_resolution = None
            if client is not None:
                client.last_reasoning = None

            try:
                if args.oracle_selectors:
                    chain = relation_chain or (
                        [edge[1] for edge in reference_paths[0]] if reference_paths else []
                    )
                    def relations_selector(question, entity, available, width, depth):
                        relation = chain[depth - 1] if depth <= len(chain) else None
                        return [
                            (rel, direction, 1.0)
                            for rel, direction in available
                            if rel == relation and direction == "outgoing"
                        ][:width]

                    def edge_selector(question, candidates, width, context):
                        depth = context["depth"] - 1
                        gold_edges = {
                            tuple(path[depth])
                            for path in reference_paths
                            if depth < len(path)
                        }
                        viable = [(edge, score) for edge, score in candidates if edge in gold_edges]
                        return (viable or list(candidates))[:width]

                    stop = lambda question, paths, depth: depth >= len(chain)
                else:
                    def relations_selector(question, entity, available, width, depth):
                        return client.select_relations(
                            question, entity, available, width, depth=depth
                        )

                    def edge_selector(question, candidates, width, context):
                        selected = client.select_edges(
                            question, candidates, width, context=context
                        )
                        return selected

                    stop = None if args.disable_early_stop else client.is_sufficient

                result = run_tog_search(
                    question,
                    [start],
                    graph,
                    relations_selector,
                    edge_selector,
                    width=args.width,
                    max_depth=args.max_depth,
                    bidirectional=args.bidirectional,
                    should_stop=stop,
                    neighborhood_threshold=args.neighborhood_threshold,
                    num_retain_entity=args.num_retain_entity,
                    seed=args.seed,
                )
                if client is not None:
                    decision = client.finish(question, result)
                    generated_answer = decision["answer"]
                    predicted_answer_entity = client.resolve_answer(generated_answer)
                    entity_resolution = client.last_entity_resolution
                    answer_reasoning = decision["reasoning"]
                    answer_source = decision["source"]
                    answer_parsing = decision["answer_parsing"]
                else:
                    predicted_answer_entity = result.frontier[0] if result.frontier else None
                    answer_source = "oracle"
                    if predicted_answer_entity is not None:
                        generated_answer = entity_title.get(predicted_answer_entity, predicted_answer_entity)
            except Exception as exc:
                if args.fail_fast:
                    raise
                failures += 1
                error = f"{type(exc).__name__}: {exc}"
                if result is None:
                    result = ToGSearchResult(
                        frontier=[], paths=[], explored_by_depth=[], termination_reason="error",
                    )

            elapsed = time.perf_counter() - started
            episode_calls = client.calls[call_start:] if client is not None else []
            top1_entity = result.frontier[0] if result.frontier else None
            top1_path = result.paths[0] if result.paths else []
            top1_score = (
                score_single_final_entity(top1_entity, gold)
                if top1_entity is not None
                else {"Hits1": 0.0, "MRR": None, "final_entity_correct": 0.0}
            )
            answer_score = (
                score_single_final_entity(predicted_answer_entity, gold)
                if predicted_answer_entity is not None
                else {"Hits1": 0.0, "MRR": None, "final_entity_correct": 0.0}
            )
            # Text answers remain scoreable even if absent from the local graph.
            text_correct = answer_text_correct(generated_answer, get_row_value(row, "Answer"))
            if text_correct:
                answer_score = {"Hits1": 1.0, "MRR": None, "final_entity_correct": 1.0}
            beam_score = score_answer_set(result.frontier, gold)
            path_score = best_path_fidelity_score(top1_path, reference_paths, relation_chain)
            path_validation = (
                validate_executed_path(top1_path, start, top1_entity, all_triplets)
                if not args.bidirectional
                else {"valid": None, "reason": "directed validator unavailable for bidirectional paths"}
            )
            generated_correct = answer_text_correct(
                generated_answer, get_row_value(row, "Answer")
            )
            actual_calls = len(episode_calls)
            prompt_tokens = sum_call_field(episode_calls, "prompt_tokens")
            completion_tokens = sum_call_field(episode_calls, "response_tokens")
            total_tokens = sum_call_field(episode_calls, "total_tokens")
            api_retries = sum(int(call.get("attempt", 0) > 0) for call in episode_calls)

            generated_answer_scores.append(answer_score)
            beam_scores.append(beam_score)
            if path_score is not None:
                path_scores.append(path_score)
            wall_times.append(elapsed)
            calls_per_question.append(actual_calls)
            tokens_per_question.append(total_tokens)

            episode = {
                "question_index": get_row_value(row, "Question-Number", row_position),
                "row_position": row_position,
                "dataframe_index": get_row_value(row, "dataframe_index"),
                "question": question,
                "dataset": args.dataset,
                "hop_split": args.hops,
                "hops": get_row_value(row, "Hops", args.hops),
                "start_entity": start,
                "start_entity_label": entity_title.get(start, start),
                "gold_answer_entities": sorted(gold),
                "gold_answer_labels": [entity_title.get(entity, entity) for entity in sorted(gold)],
                "gold_answer_text": get_row_value(row, "Answer"),
                "gold_reference_path_source": reference_source,
                "gold_reference_path_count": len(reference_paths),
                "predicted_terminal_entity": top1_entity,
                "predicted_terminal_label": entity_title.get(top1_entity, top1_entity) if top1_entity else None,
                "predicted_answer_entity": predicted_answer_entity,
                "predicted_answer_label": entity_title.get(predicted_answer_entity, predicted_answer_entity),
                "answer_source": answer_source,
                "answer_reasoning": answer_reasoning,
                "answer_parsing": answer_parsing,
                "entity_resolution": entity_resolution,
                "formatting_events": [
                    {"call_index": i, "stage": call["stage"], "kind": key, **call[key]}
                    for i, call in enumerate(episode_calls)
                    for key in ("parsing", "answer_parsing") if key in call
                ],
                "answer_score": answer_score,
                "answer_correct": bool(answer_score["Hits1"]),
                "terminal_entity_correct": bool(top1_score["Hits1"]),
                "final_entity_score": top1_score,
                "beam_entities": result.frontier,
                "beam_labels": [entity_title.get(entity, entity) for entity in result.frontier],
                "beam_width_actual": len(result.frontier),
                "beam_answer_hit": bool(beam_score["Hits1"]),
                "beam_answer_score": beam_score,
                "generated_answer": generated_answer,
                "generated_answer_correct": bool(answer_score["Hits1"]),
                "generated_answer_text_correct": generated_correct,
                "termination_reason": result.termination_reason,
                "navigation_status": "error" if error else "success",
                "status_message": error,
                "executed_path": top1_path,
                "readable_executed_path": translate_path(top1_path, entity_title, relation_title),
                "beam_paths": result.paths,
                "readable_beam_paths": [
                    translate_path(path, entity_title, relation_title) for path in result.paths
                ],
                "explored_by_depth": result.explored_by_depth,
                "neighborhood_events": result.neighborhood_events,
                "neighborhoods_sampled": sum(
                    int(event["sampling_applied"]) for event in result.neighborhood_events
                ),
                "search_depth": len(result.explored_by_depth),
                "retained_graph_edges": sum(len(edges) for edges in result.explored_by_depth),
                "path_fidelity": path_score,
                "path_validation": path_validation,
                "graph_directionality": config["graph_directionality"],
                "actual_llm_calls": actual_calls,
                "api_retries": api_retries,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "response_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "elapsed_time": elapsed,
                "model_calls": episode_calls,
                "calls_by_stage": summarize_calls_by_stage(episode_calls),
                "raw_model_outputs": [call.get("raw_output") for call in episode_calls],
            }
            episodes.append(episode)

            beam_correct = sum(score["Hits1"] for score in beam_scores)
            generated_correct = sum(
                score["Hits1"] for score in generated_answer_scores
            )

            progress.set_description(
                f"ToG answer={int(generated_correct)}/{len(generated_answer_scores)} "
                f"beam={int(beam_correct)}/{len(beam_scores)}"
            )

            statistics = {
                "questions": len(episodes),
                "failures": failures,
                "formatting_counts": {
                    status: sum(event["status"] == status for ep in episodes for event in ep["formatting_events"])
                    for status in ("strict", "tolerant", "rejected", "fallback")
                },
                "generated_answer": aggregate_answer_metrics(generated_answer_scores),
                "terminal_entity": aggregate_answer_metrics([
                    ep["final_entity_score"] for ep in episodes
                ]),
                "beam": aggregate_answer_metrics(beam_scores),
                "path_fidelity": aggregate_single_prediction_metrics(path_scores),
                "answer_sources": {
                    source: sum(ep["answer_source"] == source for ep in episodes)
                    for source in sorted({ep["answer_source"] for ep in episodes if ep["answer_source"]})
                },
                "generated_answer_correct": sum(
                    int(episode["generated_answer_correct"]) for episode in episodes
                ),
                "actual_llm_calls_total": sum(calls_per_question),
                "actual_llm_calls_per_question": mean(calls_per_question),
                "calls_by_stage": summarize_calls_by_stage([
                    call
                    for episode in episodes
                    for call in episode["model_calls"]
                ]),
                "prompt_tokens_total": sum(
                    episode["prompt_tokens"] for episode in episodes
                ),
                "completion_tokens_total": sum(
                    episode["completion_tokens"] for episode in episodes
                ),
                "total_tokens": sum(tokens_per_question),
                "tokens_per_question": mean(tokens_per_question),
                "wall_time_total": sum(wall_times),
                "wall_time_per_question": mean(wall_times),
                "retained_graph_edges_total": sum(
                    episode["retained_graph_edges"] for episode in episodes
                ),
                "neighborhoods_total": sum(
                    len(episode["neighborhood_events"]) for episode in episodes
                ),
                "neighborhoods_sampled": sum(
                    episode["neighborhoods_sampled"] for episode in episodes
                ),
            }
            if len(episodes) % args.checkpoint_every == 0 or len(episodes) == len(qa):
                write_payload(output, config, statistics, episodes)

    grapher.clear_relation_index()
    print(
        f"Generated-answer accuracy: "
        f"{statistics['generated_answer']['correct']}/{statistics['questions']} | "
        f"Terminal Hits@1: "
        f"{statistics['terminal_entity']['correct']}/{statistics['questions']} | "
        f"Beam Hits@{args.width}: "
        f"{statistics['beam']['correct']}/{statistics['questions']}"
    )
    print(
        f"LLM calls: {statistics['actual_llm_calls_total']} "
        f"({statistics['actual_llm_calls_per_question']:.2f}/question) | "
        f"Wall time: {statistics['wall_time_total']:.2f}s "
        f"({statistics['wall_time_per_question']:.2f}s/question)"
    )
    print(f"Results saved to {output}")


if __name__ == "__main__":
    main()

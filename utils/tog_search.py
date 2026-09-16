"""Local-graph search primitives for Think-on-Graph (ToG)."""

from collections import defaultdict
from dataclasses import dataclass, field
import hashlib
from random import Random
from typing import Callable, Iterable, Sequence

from utils.kgqa_types import EntityId, RelationId, Triplet, TripletCollection, TripletList


DirectionalIndex = dict[EntityId, dict[RelationId, TripletList]]
RelationSelector = Callable[
    [str, EntityId, Sequence[tuple[RelationId, str]], int, int],
    Sequence[tuple[RelationId, str, float]],
]
EdgeSelector = Callable[
    [str, Sequence[tuple[Triplet, float]], int, dict],
    Sequence[tuple[Triplet, float]],
]
StopPredicate = Callable[[str, Sequence[TripletList], int], bool]


@dataclass
class ToGSearchResult:
    frontier: list[EntityId]
    paths: list[TripletList]
    explored_by_depth: list[TripletList]
    stopped_early: bool = False
    termination_reason: str = "max_depth"
    neighborhood_events: list[dict] = field(default_factory=list)


class LocalToGGraph:
    """Bidirectional relation-aware index over a dataset graph snapshot."""

    def __init__(self, triplets: TripletCollection) -> None:
        outgoing = defaultdict(lambda: defaultdict(list))
        incoming = defaultdict(lambda: defaultdict(list))
        for raw_edge in triplets:
            edge = tuple(str(part) for part in raw_edge)
            head, relation, tail = edge
            outgoing[head][relation].append(edge)
            incoming[tail][relation].append(edge)
        self.outgoing = self._freeze(outgoing)
        self.incoming = self._freeze(incoming)

    @staticmethod
    def _freeze(index) -> DirectionalIndex:
        return {
            entity: {
                relation: sorted(edges)
                for relation, edges in sorted(relations.items())
            }
            for entity, relations in index.items()
        }

    def relations(self, entity: EntityId, bidirectional: bool = True) -> list[tuple[RelationId, str]]:
        choices = [(relation, "outgoing") for relation in self.outgoing.get(entity, {})]
        if bidirectional:
            choices.extend((relation, "incoming") for relation in self.incoming.get(entity, {}))
        return sorted(set(choices), key=lambda item: (item[0], item[1]))

    def edges(self, entity: EntityId, relation: RelationId, direction: str) -> TripletList:
        if direction == "outgoing":
            return list(self.outgoing.get(entity, {}).get(relation, []))
        if direction == "incoming":
            return list(self.incoming.get(entity, {}).get(relation, []))
        raise ValueError(f"Unknown graph direction: {direction}")

    @staticmethod
    def destination(entity: EntityId, edge: Triplet, direction: str) -> EntityId:
        if direction == "outgoing" and edge[0] == entity:
            return edge[2]
        if direction == "incoming" and edge[2] == entity:
            return edge[0]
        raise ValueError(f"Edge {edge!r} is not a legal {direction} edge for {entity!r}")


def _sampling_seed(
    seed: int,
    question: str,
    depth: int,
    entity: EntityId,
    relation: RelationId,
    direction: str,
) -> int:
    payload = f"{seed}\0{question}\0{depth}\0{entity}\0{relation}\0{direction}"
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")


def _sample_neighborhood(
    edges: TripletList,
    *,
    threshold: int | None,
    retain: int,
    seed: int,
    question: str,
    depth: int,
    entity: EntityId,
    relation: RelationId,
    direction: str,
) -> TripletList:
    """Apply original-ToG-style random neighbor retention reproducibly."""
    if threshold is None or len(edges) < threshold:
        return list(edges)
    sample_size = min(retain, len(edges))
    rng = Random(_sampling_seed(seed, question, depth, entity, relation, direction))
    return rng.sample(list(edges), sample_size)


def run_tog_search(
    question: str,
    start_entities: Iterable[EntityId],
    graph: LocalToGGraph,
    relation_selector: RelationSelector,
    edge_selector: EdgeSelector,
    *,
    neighborhood_threshold: int | None,
    num_retain_entity: int,
    seed: int,
    width: int = 3,
    max_depth: int = 3,
    bidirectional: bool = True,
    should_stop: StopPredicate | None = None,
) -> ToGSearchResult:
    """Run relation prune, per-relation entity score, global beam prune.

    The stop callback receives ALL retained triples grouped by depth, not only
    surviving paths. Scores are local to a step, as in upstream ToG.
    """
    if width < 1:
        raise ValueError("width must be positive")
    if max_depth < 0:
        raise ValueError("max_depth must be non-negative")
    if neighborhood_threshold is not None and neighborhood_threshold < 1:
        raise ValueError("neighborhood_threshold must be positive or None")
    if num_retain_entity < 1:
        raise ValueError("num_retain_entity must be positive")

    frontier = [(str(entity), [], 1.0) for entity in dict.fromkeys(start_entities)]
    if not frontier:
        return ToGSearchResult([], [], [], termination_reason="no_start_entities")
    explored_by_depth = []
    neighborhood_events = []
    previous_relations = set()
    previous_directions = {}

    for depth in range(1, max_depth + 1):
        scored_candidates = []
        # Upstream expands each unique topic entity, after retaining beam triples.
        expanded = set()
        for entity, path, path_score in frontier:
            if entity in expanded:
                continue
            expanded.add(entity)
            available = graph.relations(entity, bidirectional=bidirectional)
            previous_direction = previous_directions.get(entity)
            available = [
                (relation, direction) for relation, direction in available
                if not (relation in previous_relations
                        and previous_direction is not None
                        and direction != previous_direction)
            ]
            if not available:
                continue
            legal_relations = set(available)

            selected_relations = relation_selector(question, entity, available, width, depth)
            for relation, direction, relation_score in selected_relations:
                if (relation, direction) not in legal_relations:
                    continue
                relation_edges = graph.edges(entity, relation, direction)
                sampled_edges = _sample_neighborhood(
                    relation_edges,
                    threshold=neighborhood_threshold,
                    retain=num_retain_entity,
                    seed=seed,
                    question=question,
                    depth=depth,
                    entity=entity,
                    relation=relation,
                    direction=direction,
                )
                context = {
                    "depth": depth,
                    "entity": entity,
                    "relation": relation,
                    "direction": direction,
                    "neighbors_available": len(relation_edges),
                    "neighbors_sampled": len(sampled_edges),
                    "sampling_applied": len(sampled_edges) < len(relation_edges),
                    "neighborhood_threshold": neighborhood_threshold,
                    "num_retain_entity": num_retain_entity,
                }
                neighborhood_events.append(context)
                if not sampled_edges:
                    continue

                prior = float(relation_score)
                scored_edges = edge_selector(
                    question,
                    [(edge, prior) for edge in sampled_edges],
                    len(sampled_edges),
                    context,
                )
                candidate_by_edge = {edge: graph.destination(entity, edge, direction) for edge in sampled_edges}
                for edge, score in scored_edges:
                    if edge in candidate_by_edge and float(score) > 0:
                        scored_candidates.append(
                            (float(score), edge, path, candidate_by_edge[edge], direction)
                        )

        if not scored_candidates:
            return ToGSearchResult(
                [entity for entity, _, _ in frontier],
                [path for _, path, _ in frontier],
                explored_by_depth,
                termination_reason="no_candidates",
                neighborhood_events=neighborhood_events,
            )

        # Original ToG globally retains the width highest combined
        # relation/entity scores after scoring within each relation.
        scored_candidates.sort(key=lambda item: -item[0])
        next_frontier = []
        depth_edges = []
        previous_relations = set()
        previous_directions = {}
        for score, edge, previous_path, destination, direction in scored_candidates:
            previous_relations.add(edge[1])
            previous_directions.setdefault(destination, direction)
            depth_edges.append(edge)
            next_frontier.append((destination, previous_path + [edge], score))
            if len(next_frontier) >= width:
                break

        frontier = next_frontier
        explored_by_depth.append(depth_edges)
        paths = [path for _, path, _ in frontier]
        if should_stop is not None and should_stop(question, explored_by_depth, depth):
            return ToGSearchResult(
                [entity for entity, _, _ in frontier],
                paths,
                explored_by_depth,
                stopped_early=True,
                termination_reason="sufficient_knowledge",
                neighborhood_events=neighborhood_events,
            )

    return ToGSearchResult(
        [entity for entity, _, _ in frontier],
        [path for _, path, _ in frontier],
        explored_by_depth,
        neighborhood_events=neighborhood_events,
    )

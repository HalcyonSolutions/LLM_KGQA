import os
import sys
import unittest

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from utils.tog_search import LocalToGGraph, run_tog_search


class LocalToGSearchTests(unittest.TestCase):
    def setUp(self):
        self.graph = LocalToGGraph({
            ("book", "author", "writer"),
            ("writer", "spouse", "partner"),
            ("partner", "language", "English"),
            ("book", "genre", "novel"),
            ("other", "mentions", "book"),
        })

    def test_local_graph_exposes_directional_relations(self):
        self.assertIn(("author", "outgoing"), self.graph.relations("book"))
        self.assertIn(("mentions", "incoming"), self.graph.relations("book"))
        self.assertNotIn(("mentions", "incoming"), self.graph.relations("book", False))

    def test_beam_search_follows_only_legal_selected_edges(self):
        wanted = {"book": "author", "writer": "spouse", "partner": "language"}
        result = run_tog_search(
            "What language does the author's spouse use?",
            ["book"], self.graph,
            lambda question, entity, available, width, depth: [(wanted[entity], "outgoing", 1.0)],
            lambda question, candidates, width, context: list(candidates)[:width],
            width=2, max_depth=3,
        neighborhood_threshold=20, num_retain_entity=5, seed=42,
        )
        self.assertEqual(result.frontier, ["English"])
        self.assertEqual([edge[1] for edge in result.paths[0]], ["author", "spouse", "language"])

    def test_illegal_llm_choices_are_ignored(self):
        result = run_tog_search(
            "question", ["book"], self.graph,
            lambda question, entity, available, width, depth: [("invented", "outgoing", 1.0)],
            lambda question, candidates, width, context: candidates,
            max_depth=1,
        neighborhood_threshold=20, num_retain_entity=5, seed=42,
        )
        self.assertEqual(result.termination_reason, "no_candidates")
        self.assertEqual(result.frontier, ["book"])

    def test_sufficiency_can_stop_search_early(self):
        result = run_tog_search(
            "question", ["book"], self.graph,
            lambda question, entity, available, width, depth: [("author", "outgoing", 1.0)],
            lambda question, candidates, width, context: candidates,
            max_depth=3,
        neighborhood_threshold=20, num_retain_entity=5, seed=42,
            should_stop=lambda question, paths, depth: depth == 1,
        )
        self.assertTrue(result.stopped_early)
        self.assertEqual(result.termination_reason, "sufficient_knowledge")


if __name__ == "__main__":
    unittest.main()

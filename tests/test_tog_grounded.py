"""Regression checks for ToG search semantics and original answer/fallback behavior."""
import unittest
from unittest.mock import Mock

from model.tog_llm_client import ToGLLMKGQAClient
from utils.tog_search import LocalToGGraph, run_tog_search


class OriginalToGTests(unittest.TestCase):
    def search(self, graph, relations, edges=None, **kwargs):
        return run_tog_search(
            'question', ['start'], LocalToGGraph(graph), relations,
            edges or (lambda question, candidates, width, context: candidates),
            neighborhood_threshold=20, num_retain_entity=5, seed=42, **kwargs,
        )

    def test_current_step_scores_and_accumulated_evidence(self):
        evidence = []
        def relations(question, entity, available, width, depth):
            scores = {'start': [('a', 'outgoing', .9), ('b', 'outgoing', .1)],
                      'left': [('c', 'outgoing', .1)],
                      'right': [('d', 'outgoing', .9)]}
            return scores.get(entity, [])
        result = self.search(
            {('start', 'a', 'left'), ('start', 'b', 'right'),
             ('left', 'c', 'x'), ('right', 'd', 'y')}, relations,
            width=2, max_depth=2,
            should_stop=lambda q, history, depth: evidence.append([list(x) for x in history]) or False,
        )
        self.assertEqual(result.frontier[0], 'y')
        self.assertEqual(len(evidence[-1]), 2)
        self.assertIn(('start', 'a', 'left'), evidence[-1][0])

    def test_pruned_branch_remains_in_reasoning_evidence(self):
        evidence = []
        def relations(q, entity, available, width, depth):
            return [(r, d, 1.) for r, d in available if d == 'outgoing']
        result = self.search(
            {('start', 'a', 'dead'), ('start', 'b', 'live'), ('live', 'c', 'end')},
            relations, width=2, max_depth=2,
            should_stop=lambda q, history, depth: evidence.append([list(x) for x in history]) or False,
        )
        self.assertEqual(result.frontier, ['end'])
        self.assertIn(('start', 'a', 'dead'), evidence[-1][0])

    def test_duplicate_destinations_retain_distinct_triples(self):
        result = self.search(
            {('start', 'a', 'end'), ('start', 'b', 'end')},
            lambda q, e, options, w, d: [(r, direction, .5) for r, direction in options],
            width=2, max_depth=1,
        )
        self.assertEqual(len(result.explored_by_depth[0]), 2)

    def test_incoming_search_and_reverse_relation_filter(self):
        seen = []
        def relations(q, entity, options, width, depth):
            seen.append((entity, options))
            return [(r, d, 1.) for r, d in options]
        result = self.search({('other', 'r', 'start'), ('other', 's', 'end')},
                             relations, max_depth=2, width=1)
        self.assertEqual(result.frontier, ['end'])
        self.assertNotIn(('r', 'outgoing'), seen[1][1])

    def test_zero_scores_do_not_expand(self):
        result = self.search({('start', 'r', 'end')},
                             lambda *args: [('r', 'outgoing', 0.)], max_depth=1)
        self.assertEqual(result.termination_reason, 'no_candidates')
        self.assertEqual(result.explored_by_depth, [])

    def client(self, outputs):
        client = object.__new__(ToGLLMKGQAClient)
        client.entity_title = {'answer': 'Answer label'}
        client.relation_title = {}
        client._label_ids = {'answer label': {'answer'}}
        client.calls = []
        client.temperature = 0
        client.exploration_temperature = .4
        client.reasoning_temperature = 0
        client.structured_output = False
        client.max_parse_retries = 0
        client.last_reasoning = None
        client.chat = Mock(side_effect=[({'message': {'content': output}}, {'status': 'success'})
                                        for output in outputs])
        client.normalize_usage = lambda response: {}
        return client

    def test_sufficient_answer_reuses_original_reasoning(self):
        from utils.tog_search import ToGSearchResult
        client = self.client(['{Yes}. The answer is {Answer label}.'])
        evidence = [[('start', 'r', 'answer')]]
        self.assertTrue(client.is_sufficient('question', evidence, 1))
        decision = client.finish('question', ToGSearchResult(['answer'], evidence, evidence, True))
        self.assertEqual(decision['answer'], 'Answer label')
        self.assertEqual(client.resolve_answer(decision['answer']), 'answer')
        self.assertEqual(client.chat.call_count, 1)

    def test_depth_limit_uses_question_only_fallback(self):
        from utils.tog_search import ToGSearchResult
        client = self.client(['The answer is {Outside graph}.'])
        result = ToGSearchResult(['answer'], [], [[('UNIQUE_SOURCE', 'r', 'answer')]])
        decision = client.finish('UNIQUE_QUESTION', result)
        self.assertEqual(decision['source'], 'knowledge_only_fallback')
        self.assertEqual(decision['answer'], 'Outside graph')
        self.assertNotIn('UNIQUE_SOURCE', client.chat.call_args.args[0])
        self.assertIn('UNIQUE_QUESTION', client.chat.call_args.args[0])
        self.assertIsNone(client.resolve_answer(decision['answer']))

    def test_dead_end_keeps_evidence_and_allows_knowledge(self):
        from utils.tog_search import ToGSearchResult
        client = self.client(['The answer is {Outside graph}.'])
        result = ToGSearchResult(['answer'], [], [[('UNIQUE_SOURCE', 'r', 'answer')]],
                                 termination_reason='no_candidates')
        decision = client.finish('question', result)
        self.assertEqual(decision['source'], 'partial_evidence_fallback')
        self.assertIn('UNIQUE_SOURCE', client.chat.call_args.args[0])
        self.assertIn('triplets and your knowledge', client.chat.call_args.args[0])

    def test_no_topics_uses_knowledge_fallback(self):
        result = run_tog_search('question', [], LocalToGGraph([]), Mock(), Mock(),
                                neighborhood_threshold=20, num_retain_entity=5, seed=42)
        client = self.client(['The answer is {Outside graph}.'])
        self.assertEqual(client.finish('question', result)['source'], 'knowledge_only_fallback')

    def test_malformed_entity_scores_use_uniform_fallback(self):
        client = self.client(['Bad scores'])
        candidates = [(('start', 'r', 'a'), .6), (('start', 'r', 'b'), .6)]
        self.assertEqual([score for _, score in client.select_edges('question', candidates, 2)], [.3, .3])
        self.assertEqual(client.calls[0]['temperature'], .4)

    def test_scores_need_not_sum_to_one_like_upstream(self):
        client = self.client(['0.2, 0.2'])
        candidates = [(('start', 'r', 'a'), .5), (('start', 'r', 'b'), .5)]
        self.assertEqual([score for _, score in client.select_edges('question', candidates, 2)], [.1, .1])

    def test_relation_parsing_prefers_outgoing(self):
        client = self.client(['1. {r [r] (Score: 0.9)}: useful'])
        selected = client.select_relations('question', 'start', [('r', 'incoming'), ('r', 'outgoing')], 3)
        self.assertEqual(selected, [('r', 'outgoing', .9)])

    def test_single_neighbor_uses_relation_score_without_llm_call(self):
        client = self.client([])
        candidates = [(('start', 'r', 'answer'), .3)]
        self.assertEqual(client.select_edges('question', candidates, 1), candidates)
        client.chat.assert_not_called()

    def test_ambiguous_label_does_not_choose_arbitrary_entity(self):
        client = self.client([])
        client._label_ids['ambiguous'] = {'a', 'b'}
        self.assertIsNone(client.resolve_answer('Ambiguous'))

    def test_runner_scores_text_answer_absent_from_graph(self):
        import contextlib
        import io
        import json
        from pathlib import Path
        import sys
        import tempfile
        from unittest.mock import patch
        import pandas as pd
        import kgqa_tog

        triples = [('start', 'r', 'answer'), ('answer', 's', 'later')]
        qa = pd.DataFrame([{
            'Question': 'question', 'Source-Entity': 'start', 'Answer-Entity': 'outside',
            'Answer': 'Outside graph', 'SplitLabel': 'test', 'Hops': 1,
            'Paths': [[triples[0]]], 'Path-Key': ['r'],
        }])
        client = self.client(['{r [r] (Score: 1.0)}', '{s [s] (Score: 1.0)}',
                              'The answer is {Outside graph}.'])
        with tempfile.TemporaryDirectory() as directory:
            argv = ['kgqa_tog.py', '--max-depth', '2', '--disable-early-stop',
                     '--result-dir', directory]
            with patch.object(sys, 'argv', argv), \
                 patch.object(kgqa_tog, 'load_triplets', return_value=pd.DataFrame(triples)), \
                 patch.object(kgqa_tog, 'load_pandas', return_value=qa), \
                 patch.object(kgqa_tog, 'load_title_maps', return_value=(client.entity_title, {}, {})), \
                 patch.object(kgqa_tog, 'ToGLLMKGQAClient', return_value=client), \
                 contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                kgqa_tog.main()
            payload = json.loads(next(Path(directory).rglob('*.json')).read_text())
        episode = payload['episodes'][0]
        self.assertEqual(episode['predicted_terminal_entity'], 'later')
        self.assertIsNone(episode['predicted_answer_entity'])
        self.assertEqual(episode['generated_answer'], 'Outside graph')
        self.assertTrue(episode['answer_correct'])
        self.assertEqual(episode['answer_source'], 'knowledge_only_fallback')
        self.assertFalse(episode['terminal_entity_correct'])
        self.assertEqual(payload['statistics']['top1']['correct'], 1)
        self.assertEqual(payload['statistics']['terminal_entity']['correct'], 0)


if __name__ == '__main__':
    unittest.main()

import unittest
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from utils.tog_parsing import parse_relations, parse_answer, parse_sufficiency


class FormattingTests(unittest.TestCase):
    def test_relation_formats(self):
        options = [('P27', 'outgoing')]
        labels = {'P27': 'country of citizenship'}
        for text in ['{country of citizenship [P27] (Score: 1.0)}',
                     '**{country of citizenship (P27) (Score: 1.0)}**',
                     '**`country of citizenship [P27]` (Score: 1.0)**',
                     '{country of citizenship [P27]} (Score: 1.0)',
                     'country of citizenship (Score: 1.0)']:
            with self.subTest(text=text):
                choices, log = parse_relations(text, options, labels)
                self.assertEqual(choices, [('P27', 'outgoing', 1.)])
                self.assertIn(log['status'], ['strict', 'tolerant'])

    def test_unknown_ambiguous_and_unscored_relations(self):
        options = [('P27', 'outgoing'), ('P17', 'outgoing')]
        labels = {'P27': 'country', 'P17': 'country'}
        for text in ['country (P999) (Score: 1)', 'country (Score: 1)',
                     'P27 or P17 (Score: 1)', 'P27 is relevant', 'P27 (Score: -1)']:
            self.assertEqual(parse_relations(text, options, labels)[0], [])

    def test_duplicate_relation_not_expanded_twice(self):
        self.assertEqual(len(parse_relations('P27 (Score: 1)\nP27 (Score: 1)',
                                             [('P27', 'outgoing')], {})[0]), 1)

    def test_multiple_relations_per_line(self):
        options = [('P108', 'outgoing'), ('P106', 'incoming')]
        labels = {'P108': 'employer', 'P106': 'occupation'}
        parts = ['{employer [P108] (Score: 0.6)}',
                 '{occupation [P106] (Score: 0.4)}']
        for separator in ['\n', '; ', ' ']:
            with self.subTest(separator=separator):
                choices, log = parse_relations(separator.join(parts), options, labels)
                self.assertEqual(choices, [('P108', 'outgoing', 0.6),
                                           ('P106', 'incoming', 0.4)])
                self.assertEqual(log['status'], 'strict')
                self.assertEqual(len(log['selections']), 2)
                self.assertEqual(log['selections'][1]['raw_fragment'], parts[1])

    def test_inline_selections_keep_validation(self):
        options = [('P108', 'outgoing'), ('P106', 'incoming')]
        labels = {'P108': 'employer', 'P106': 'occupation'}
        raw = ('employer (P999) (Score: 0.9); employer (Score: 0.6); '
               'P108 (Score: 0.2); P106 (Score: -1); '
               '**occupation [P106]** (Score: 0.4)')
        choices, log = parse_relations(raw, options, labels)
        self.assertEqual(choices, [('P108', 'outgoing', 0.6),
                                   ('P106', 'incoming', 0.4)])
        self.assertEqual([event['status'] for event in log['selections']],
                         ['rejected', 'tolerant', 'rejected', 'rejected', 'tolerant'])

    def test_answers(self):
        for raw, answer in [('{Yes}. The answer is {Dutch}.', 'Dutch'),
                            ('{Yes}. Therefore, the answer to the question is **Dutch**.', 'Dutch'),
                            ('The answer to the question is:\n**New York City**.', 'New York City'),
                            ('Therefore, the sport is **American football**.', 'American football'),
                            ('**Answer:** **University of Oxford**', 'University of Oxford'),
                            ('Dutch', 'Dutch')]:
            with self.subTest(raw=raw):
                self.assertEqual(parse_answer(raw)[0], answer)

    def test_mentions_and_ambiguity_do_not_become_answers(self):
        for text in ['The answer is {England} or {France}.',
                     'The answer is maybe Dutch.',
                     'The answer is not Dutch.',
                     'It might be **Dutch**.',
                     'Dutch is mentioned, but I cannot answer.',
                     'The sport is not **American football**.']:
            with self.subTest(text=text):
                self.assertIsNone(parse_answer(text)[0])

    def test_sufficiency_formats_and_incidental_yes(self):
        for raw in ['{Yes}. explanation', '**Yes**. explanation', 'Yes: explanation', '(Yes). explanation']:
            self.assertTrue(parse_sufficiency(raw)[0])
        for raw in ['Yesterday was sunny', 'The example says {Yes}', '{No}. explanation']:
            self.assertFalse(parse_sufficiency(raw)[0])


if __name__ == '__main__':
    unittest.main()

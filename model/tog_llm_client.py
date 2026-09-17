"""Original ToG prompting and decisions adapted to our local graph/API client."""

import re
from typing import Sequence

from model.base_llm_client import BaseLLMKGQAClient
# DO NOT use the *_wiki prompts here. They expect the original ToG
# Wikidata relation representation (e.g., wiki.relation.*) and its
# corresponding normalization/parsing logic. This adapter instead exposes
# relations as human-readable labels with Wikidata P-IDs (e.g., "country [P17]").
# Using the Wiki prompt family therefore causes relation parsing/search failures.
# Use the original ToG prompt family below.
from model.tog_original_prompts import (
    cot_prompt,
    answer_prompt,  
    extract_relation_prompt, 
    prompt_evaluate,
    score_entity_candidates_prompt,
)
from utils.kgqa_types import TripletList
from utils.tog_parsing import parse_relations, parse_answer, parse_sufficiency, audit


class ToGLLMKGQAClient(BaseLLMKGQAClient):
    def __init__(self, *args, entity_title=None, relation_title=None,
                 max_parse_retries=2, structured_output=False,
                 exploration_temperature=0.4, reasoning_temperature=0.0, **kwargs):
        if structured_output:
            raise ValueError("Original ToG uses free-text prompts; omit --structured-output")
        if max_parse_retries < 0:
            raise ValueError("max_parse_retries must be non-negative")
        super().__init__(*args, **kwargs)
        self.entity_title = entity_title or {}
        self.relation_title = relation_title or {}
        self.max_parse_retries = max_parse_retries
        self.structured_output = False
        self.exploration_temperature = exploration_temperature
        self.reasoning_temperature = reasoning_temperature
        self.calls = []
        self.last_reasoning = None
        self._label_ids = {}
        for entity, label in self.entity_title.items():
            self._label_ids.setdefault(self.normalize_answer(label), set()).add(entity)

    @staticmethod
    def normalize_answer(text):
        return str(text).strip().strip(' "\'.').casefold()

    def resolve_answer(self, answer):
        """Resolve an unambiguous exact ID/label, independently of gold or retrieval."""
        self.last_entity_resolution = audit('rejected', 'entity_resolution', reason='no_unique_exact_match')
        if answer is None:
            return None
        if answer in self.entity_title:
            self.last_entity_resolution = audit('strict', 'exact_entity_id', entity_id=answer)
            return answer
        matches = self._label_ids.get(self.normalize_answer(answer), set())
        if len(matches) == 1:
            entity = next(iter(matches))
            self.last_entity_resolution = audit('strict', 'unique_exact_label', entity_id=entity)
            return entity
        decorated = re.fullmatch(r'(.*?)\s*[\[(]([\w.-]+)[\])]\s*', answer)
        if decorated:
            label, entity = decorated.groups()
            if entity in self.entity_title and self.normalize_answer(label) == self.normalize_answer(self.entity_title[entity]):
                self.last_entity_resolution = audit('tolerant', 'label_with_entity_id', entity_id=entity)
                return entity
        return None

    def _call_text(self, prompt, stage, **metadata):
        temperature = (self.exploration_temperature
                       if stage in {'relation_selection', 'entity_scoring'}
                       else self.reasoning_temperature)
        last_error = None
        for attempt in range(self.max_parse_retries + 1):
            previous_temperature = self.temperature
            try:
                self.temperature = temperature
                response, status = self.chat(prompt, response_format=None)
            finally:
                self.temperature = previous_temperature
            status = dict(status)
            status.update(self.normalize_usage(response))
            content = response.get('message', {}).get('content') if isinstance(response, dict) else None
            self.calls.append({**status, **metadata, 'stage': stage, 'attempt': attempt,
                               'raw_output': content, 'structured_output': False,
                               'temperature': temperature})
            if status.get('status') == 'success' and isinstance(content, str):
                return content
            last_error = status.get('message', 'Missing successful text response')
        raise RuntimeError(f'ToG API failed after {self.max_parse_retries + 1} attempts: {last_error}')

    def _relation_name(self, relation):
        # IDs disambiguate equal local labels; upstream Freebase names are unique.
        return f'{self.relation_title.get(relation, relation)} [{relation}]'

    def select_relations(self, question, entity, available, width, depth=None):
        by_name = {}
        # Original Freebase parser prefers outgoing when a relation exists both ways.
        for relation, direction in sorted(available, key=lambda item: item[1] != 'outgoing'):
            by_name.setdefault(self._relation_name(relation), (relation, direction))
        prompt = (extract_relation_prompt % (width, width) + question
                  + '\nTopic Entity: ' + self.entity_title.get(entity, entity)
                  + '\nRelations: ' + '; '.join(sorted(by_name)) + '\nA: ')
        raw = self._call_text(prompt, 'relation_selection', entity=entity, depth=depth,
                              available_options=len(by_name), requested_width=width)
        selected, parsing = parse_relations(raw, list(by_name.values()), self.relation_title)
        self.calls[-1]['parsing'] = parsing
        return selected

    def select_edges(self, question, candidates, width, context=None):
        if len(candidates) <= 1:
            return list(candidates)
        context = context or {}
        direction = context.get('direction', 'outgoing')
        endpoint = 2 if direction == 'outgoing' else 0
        ordered = sorted(candidates, key=lambda item: (
            self.entity_title.get(item[0][endpoint], item[0][endpoint]), item[0][endpoint]))
        relation = context.get('relation', ordered[0][0][1])
        names = [self.entity_title.get(edge[endpoint], edge[endpoint]) for edge, _ in ordered]
        prompt = (score_entity_candidates_prompt.format(question, self._relation_name(relation))
                  + '; '.join(names) + '\nScore: ')
        raw = self._call_text(prompt, 'entity_scoring', available_options=len(ordered),
                              requested_width=width, **context)
        scores = [float(value) for value in re.findall(r'\d+\.\d+', raw)]
        # Upstream does not enforce normalization; malformed counts use equal scores.
        fallback = len(scores) != len(ordered)
        if fallback:
            scores = [1 / len(ordered)] * len(ordered)
        self.calls[-1]['parsing'] = audit('fallback' if fallback else 'strict',
                                          'uniform_scores' if fallback else 'decimal_scores')
        return [(edge, score * prior) for (edge, prior), score in zip(ordered, scores)]

    def _evidence(self, evidence_by_depth):
        return '\n'.join(', '.join(str((self.entity_title.get(head, head),
                                        self.relation_title.get(relation, relation),
                                        self.entity_title.get(tail, tail)))
                                   for head, relation, tail in edges)
                         for edges in evidence_by_depth)

    @staticmethod
    def extract_answer(raw):
        return parse_answer(raw)[0]

    def _decision(self, raw, source, sufficient=False):
        answer, parsing = parse_answer(raw)
        self.calls[-1]['answer_parsing'] = parsing
        return {'reasoning': raw, 'answer': answer, 'answer_parsing': parsing,
                'sufficient': sufficient, 'source': source}

    def is_sufficient(self, question: str, evidence_by_depth: Sequence[TripletList], depth: int):
        prompt = (prompt_evaluate + question + '\nKnowledge Triplets: '
                  + self._evidence(evidence_by_depth) + 'A: ')
        raw = self._call_text(prompt, 'reasoning', depth=depth,
                              evidence_edges=sum(map(len, evidence_by_depth)))
        sufficient, parsing = parse_sufficiency(raw)
        self.calls[-1]['parsing'] = parsing
        self.last_reasoning = self._decision(raw, 'reasoning', sufficient)
        return sufficient

    def generate_answer(self, question, evidence_by_depth):
        prompt = (answer_prompt + question + '\n\nKnowledge Triplets: '
                  + self._evidence(evidence_by_depth) + 'A: ')
        raw = self._call_text(prompt, 'answer_generation',
                              evidence_edges=sum(map(len, evidence_by_depth)))
        return self._decision(raw, 'partial_evidence_fallback')

    def generate_without_explored_paths(self, question):
        raw = self._call_text(cot_prompt + '\n\nQ: ' + question + '\nA:',
                              'knowledge_only_fallback', evidence_edges=0)
        return self._decision(raw, 'knowledge_only_fallback')

    def finish(self, question, result):
        if result.stopped_early:
            return self.last_reasoning
        if result.termination_reason in {'max_depth', 'no_start_entities'}:
            return self.generate_without_explored_paths(question)
        return self.generate_answer(question, result.explored_by_depth)

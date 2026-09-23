"""Replay answer extraction without changing saved runs or simulating new navigation."""
import argparse
from collections import Counter
import json
from pathlib import Path

import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from kgqa_tog import answer_text_correct
from utils.tog_parsing import PARSER_VERSION, parse_answer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('result')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if Path(args.result).resolve() == Path(args.output).resolve():
        raise ValueError('Audit output must not overwrite the original result')
    payload = json.loads(Path(args.result).read_text())
    rows = []
    for episode in payload['episodes']:
        raw = episode.get('answer_reasoning')
        if raw is None:
            raw = episode.get('generated_answer') or ''
        answer, parsing = parse_answer(raw)
        # Gold is used only AFTER extraction, never to choose an answer span.
        correct = answer_text_correct(answer, episode.get('gold_answer_text'))
        rows.append(dict(question_index=episode.get('question_index'),
                         previous_answer=episode.get('generated_answer'),
                         previous_text_correct=episode.get('generated_answer_text_correct', False),
                         extracted_answer=answer, text_correct=correct, parsing=parsing))
    result = dict(source=str(Path(args.result)), parser_version=PARSER_VERSION,
                  scope='offline_answer_text_extraction_only_original_navigation',
                  questions=len(rows), text_correct=sum(row['text_correct'] for row in rows),
                  parsing_counts=dict(Counter(row['parsing']['status'] for row in rows)), episodes=rows)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2))
    print(json.dumps({k:v for k,v in result.items() if k != 'episodes'}, indent=2))


if __name__ == '__main__':
    main()

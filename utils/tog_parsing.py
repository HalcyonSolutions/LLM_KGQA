"""Auditable formatting tolerance for ToG; no gold answers enter these parsers."""
import re

PARSER_VERSION = 'tog_format_v2'


def strip_markup(text):
    return re.sub(r'(\*\*|__|`)(.*?)\1', r'\2', text, flags=re.S)


def clean(text):
    text = strip_markup(text).strip()
    return text.strip(' \t\r\n"\'{}[]').rstrip('.').strip()


def audit(status, method, **details):
    return dict(parser_version=PARSER_VERSION, status=status, method=method, **details)


def parse_relations(raw, options, labels):
    """Accept scored selections only, resolved against available relation IDs."""
    names = {f'{labels.get(r, r)} [{r}]': (r, d) for r, d in reversed(options)}
    selected, events, seen = [], [], set()
    # Process each scored selection, including multiple selections on one line.
    fragments = re.finditer(
        r'[^;\n]*?\bScore\s*:\s*[+-]?(?:\d+(?:\.\d*)?|\.\d+)'
        r'[^\S\n]*\)?[^\S\n]*\}?', raw, re.I,
    )
    for fragment in fragments:
        line = fragment.group().strip()
        score_match = re.search(r'\bScore\s*:\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))', line, re.I)
        if not score_match:
            continue
        score = float(score_match[1])
        prefix = line[:score_match.start()]
        strict = re.search(r'\{\s*(.*?)\s+\(Score:\s*([0-9.]+)\)\s*\}', line)
        choice = names.get(strict[1].strip()) if strict else None
        method = 'original_syntax'
        if choice is None:
            method = 'available_relation_id'
            ids = {r for r, _ in options if re.search(r'(?<![\w])' + re.escape(r) + r'(?![\w])', prefix)}
            # Unknown Wikidata IDs must never silently resolve by a familiar label.
            mentioned_ids = set(re.findall(r'\bP\d+\b', prefix))
            if mentioned_ids - {r for r, _ in options}:
                ids = set()
                method = 'unknown_relation_id'
            elif not ids:
                method = 'unique_relation_label'
                name = clean(re.sub(r'^\s*\d+[.)]\s*', '', prefix)).strip(' (){}[]')
                ids = {r for r, _ in options if clean(labels.get(r, r)).casefold() == name.casefold()}
            if len(ids) == 1:
                relation = next(iter(ids))
                choice = next((r, d) for r, d in options if r == relation)
        if choice is None or not 0 <= score <= 1:
            events.append(audit('rejected', method, raw_fragment=line, reason='unknown_or_ambiguous_relation_or_invalid_score'))
            continue
        if choice in seen:
            events.append(audit('rejected', 'duplicate_selection', raw_fragment=line))
            continue
        seen.add(choice)
        selected.append((*choice, score))
        events.append(audit('strict' if method == 'original_syntax' else 'tolerant', method,
                            raw_fragment=line, relation=choice[0], direction=choice[1], score=score))
    status = 'rejected' if not selected else ('tolerant' if any(e['status'] != 'strict' for e in events) else 'strict')
    return selected, audit(status, 'scored_available_relations', selections=events,
                           reason=None if selected else 'no_accepted_scored_relation')


def parse_sufficiency(raw):
    match = re.match(r'^\s*(?:A:\s*)?(?:\*\*|`)?\s*[\{\[(]?\s*(yes|no)\s*[\}\])]?\s*(?:\*\*|`)?(?=[\s.,:!]|$)', raw, re.I)
    if not match:
        return False, audit('rejected', 'sufficiency', reason='missing_leading_yes_no')
    strict = bool(re.match(r'^\s*\{(?:Yes|No)\}', raw))
    return match[1].lower() == 'yes', audit('strict' if strict else 'tolerant', 'leading_yes_no', raw_fragment=match[0])


def parse_answer(raw):
    groups = [clean(g) for g in re.findall(r'\{([^{}]+)\}', raw)
              if clean(g).casefold() not in {'yes', 'no'}]
    unique = list(dict.fromkeys(groups))
    if len(unique) > 1:
        return None, audit('rejected', 'braced_answer', reason='multiple_distinct_answers', candidates=unique)
    if unique:
        if re.search(r'\b(?:not|maybe|perhaps|possibly)\s+\{', raw, re.I):
            return None, audit('rejected', 'braced_answer', reason='negated_or_uncertain_answer')
        normalized = any(g.strip() != clean(g) for g in re.findall(r'\{([^{}]+)\}', raw) if clean(g).casefold() not in {'yes', 'no'})
        return unique[0], audit('tolerant' if normalized else 'strict', 'braced_answer', extracted=unique[0])

    # Prefer an explicit final answer clause, including Markdown headings/newlines.
    plain = strip_markup(raw)
    matches = list(re.finditer(r'(?:\b(?:final\s+)?answer(?:\s+to\s+(?:the|your)\s+question)?\s*(?:is\s*:?|:)|^\s*#{1,6}\s*Answer\s*:?)\s*', plain, re.I | re.M))
    fragment = plain[matches[-1].end():].strip() if matches else None
    method = 'explicit_answer_clause'
    if fragment is not None:
        fragment = fragment.split('\n')[0].strip()
        # Full sentences are not entity answers; extract only a final emphasized value.
        if len(fragment.split()) > 12:
            fragment = None
    if not fragment:
        # Accept a final asserted bold value, never a mention elsewhere in the rationale.
        last = re.search(r'([^\n.!?]*?)\*\*([^*\n]+)\*\*\s*[.!]?\s*$', raw)
        if last and re.search(r'\b(?:is|was|are|were)\s*$', last[1], re.I) and not re.search(r'\b(?:not|might|maybe|perhaps|could|whether)\b', last[1], re.I):
            fragment, method = last[2], 'final_asserted_bold_value'
    if fragment:
        value = clean(fragment)
        if value and not re.search(r'\b(?:or|not|maybe|perhaps|unknown|cannot|unable)\b|\?', value, re.I):
            return value, audit('tolerant', method, extracted=value)
        return None, audit('rejected', method, reason='ambiguous_or_refused_answer')
    value = clean(raw)
    if '\n' not in value and len(value.split()) <= 8 and not re.search(r'\b(?:yes|no|not|or|cannot|unable|unknown|maybe|perhaps|might|could|whether|answer|sorry)\b|[?!]', value, re.I):
        return value, audit('tolerant', 'bare_answer', extracted=value)
    return None, audit('rejected', 'answer', reason='no_unambiguous_answer_span')

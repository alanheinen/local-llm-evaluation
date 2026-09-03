"""IFEval instruction verifiers.

IFEval measures whether a model FOLLOWS a verifiable instruction - "write in
all lowercase", "use exactly 3 bullet points", "wrap the answer in quotes" -
rather than whether the answer is good. Every check is programmatic, which is
what makes it worth having on a local rig: no judge model is required, so it
produces a real number on a machine with no API budget.

WE IMPLEMENT A SUBSET, AND THE IMPORTER DROPS THE REST.

The official suite defines ~25 instruction types. Implementing some and
silently scoring the others as failures would be worse than not running it at
all, because a model would be marked wrong for an instruction nobody checked.
So VERIFIERS is the contract: the importer keeps only cases whose every
instruction appears here, and records how many it dropped. A model's score is
then over instructions that were genuinely checked.

Semantics follow google-research/instruction_following_eval. Where this is
deliberately narrower than the original, the docstring says so.
"""
from __future__ import annotations

import re


def _words(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text)


def _paragraphs(text: str) -> list[str]:
    return [p for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]


def _change_case_lowercase(text, kw):
    return text == text.lower(), "contains uppercase"


def _change_case_uppercase(text, kw):
    return text == text.upper(), "contains lowercase"


def _keywords_existence(text, kw):
    low = text.lower()
    missing = [k for k in (kw.get("keywords") or []) if k.lower() not in low]
    return not missing, f"missing keywords: {missing}"


def _keywords_forbidden(text, kw):
    low = text.lower()
    present = [k for k in (kw.get("forbidden_words") or []) if k.lower() in low]
    return not present, f"forbidden words present: {present}"


def _keywords_frequency(text, kw):
    word, n, rel = kw.get("keyword"), kw.get("frequency"), kw.get("relation")
    if word is None or n is None:
        return False, "verifier kwargs incomplete"
    c = len(re.findall(rf"\b{re.escape(word)}\b", text, re.I))
    if rel == "at least":
        return c >= n, f"{c} occurrences < {n}"
    if rel == "less than":
        return c < n, f"{c} occurrences >= {n}"
    return c == n, f"{c} occurrences != {n}"


def _length_words(text, kw):
    n, rel = kw.get("num_words"), kw.get("relation")
    c = len(_words(text))
    if rel == "at least":
        return c >= n, f"{c} words < {n}"
    return c <= n, f"{c} words > {n}"


def _length_sentences(text, kw):
    n, rel = kw.get("num_sentences"), kw.get("relation")
    c = len([s for s in re.split(r"[.!?]+(?:\s|$)", text.strip()) if s.strip()])
    if rel == "at least":
        return c >= n, f"{c} sentences < {n}"
    return c <= n, f"{c} sentences > {n}"


def _length_paragraphs(text, kw):
    n = kw.get("num_paragraphs")
    c = len(_paragraphs(text))
    return c == n, f"{c} paragraphs != {n}"


def _detectable_bullets(text, kw):
    n = kw.get("num_bullets")
    c = len(re.findall(r"^\s*[\*\-]\s+", text, re.M))
    return c == n, f"{c} bullets != {n}"


def _detectable_sections(text, kw):
    n = kw.get("num_sections")
    marker = kw.get("section_spliter") or kw.get("section_splitter")
    if not marker:
        return False, "verifier kwargs incomplete"
    c = len(re.findall(rf"{re.escape(marker)}\s*\d*", text, re.I))
    return c >= n, f"{c} sections < {n}"


def _detectable_highlights(text, kw):
    n = kw.get("num_highlights")
    c = len(re.findall(r"\*[^\*\n]+\*", text))
    return c >= n, f"{c} highlights < {n}"


def _detectable_title(text, kw):
    return bool(re.search(r"<<[^\n]+>>", text)), "no <<title>> found"


def _detectable_json(text, kw):
    import json
    t = re.sub(r"^\s*```(?:json)?|```\s*$", "", text.strip(), flags=re.M).strip()
    try:
        json.loads(t)
        return True, "ok"
    except Exception:  # noqa: BLE001
        return False, "not parseable as one JSON object"


def _startend_quotation(text, kw):
    t = text.strip()
    return len(t) >= 2 and t.startswith('"') and t.endswith('"'), "not wrapped in quotes"


def _startend_end_checker(text, kw):
    end = (kw.get("end_phrase") or "").strip().lower()
    return text.strip().lower().endswith(end), f"does not end with {end!r}"


def _punctuation_no_comma(text, kw):
    return "," not in text, "contains a comma"


def _combination_repeat_prompt(text, kw):
    p = (kw.get("prompt_to_repeat") or "").strip()
    return text.strip().startswith(p), "does not begin with the prompt verbatim"


def _combination_two_responses(text, kw):
    return len(re.findall(r"\*\*\*+", text)) >= 1, "no ****** separator between responses"


VERIFIERS = {
    "change_case:english_lowercase": _change_case_lowercase,
    "change_case:english_capital": _change_case_uppercase,
    "keywords:existence": _keywords_existence,
    "keywords:forbidden_words": _keywords_forbidden,
    "keywords:frequency": _keywords_frequency,
    "length_constraints:number_words": _length_words,
    "length_constraints:number_sentences": _length_sentences,
    "length_constraints:number_paragraphs": _length_paragraphs,
    "detectable_format:number_bullet_lists": _detectable_bullets,
    "detectable_format:multiple_sections": _detectable_sections,
    "detectable_format:number_highlighted_sections": _detectable_highlights,
    "detectable_format:title": _detectable_title,
    "detectable_format:json_format": _detectable_json,
    "startend:quotation": _startend_quotation,
    "startend:end_checker": _startend_end_checker,
    "punctuation:no_comma": _punctuation_no_comma,
    "combination:repeat_prompt": _combination_repeat_prompt,
    "combination:two_responses": _combination_two_responses,
}


def verify(response: str, instruction_id: str, kwargs: dict) -> tuple[bool, str]:
    """Check one instruction. An unknown id is an ERROR, never a pass.

    Returning False for an unimplemented verifier marks the model wrong for
    something nobody checked; returning True inflates every score. The importer
    prevents this case arising, so reaching here means the contract broke.
    """
    fn = VERIFIERS.get(instruction_id)
    if fn is None:
        return False, f"NO VERIFIER for {instruction_id} - this case should not have been imported"
    return fn(response or "", kwargs or {})


def verify_all(response: str, instructions: list[dict]) -> tuple[bool, str]:
    """Strict conjunction, as IFEval defines it: every instruction must hold.

    IFEval's headline metric is strict prompt-level accuracy. Partial credit
    for following three of four instructions is not what it measures, so this
    does not average.
    """
    failures = []
    for item in instructions:
        ok, why = verify(response, item.get("id"), item.get("kwargs"))
        if not ok:
            failures.append(f"{item.get('id')}: {why}")
    return (not failures), ("; ".join(failures)[:300] if failures else "ok")

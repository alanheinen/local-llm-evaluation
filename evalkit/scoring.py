"""Deterministic scoring. No judge model, by design.

Every scorer here is programmatic. That is a constraint worth keeping even if
you have an API budget: a judge introduces a second model's opinions into your
measurements, and on a local rig it is usually the largest source of variance
and cost. Everything in this kit produces a number a second person can
reproduce exactly from the stored responses.

Four case types:

  mcq        multiple choice - GPQA, MMLU-Pro
  code       execute the model's code against tests - HumanEval+
  ifeval     verifiable instruction following
  answered   did the model answer at all - XSTest over-refusal

A scorer returns None rather than 0.0 when it cannot judge the response. That
distinction carries the whole weight of the unscorable-rate reporting in
report.py: a model that produced no parseable answer has not scored zero, it
has not been measured, and averaging over the cases it did answer inflates it.
"""
from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path

from .ifeval import verify_all
from .refusal import check_answered

FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)


def extract_code(text: str) -> str:
    """Prefer fenced blocks; fall back to the whole response."""
    blocks = FENCE.findall(text or "")
    if blocks:
        return max(blocks, key=len).strip()
    return (text or "").strip()


def run_python(answer_code: str, test_code: str, timeout_s: int = 20) -> tuple[bool, str]:
    """Append the tests to the model's code and execute.

    THIS RUNS UNTRUSTED MODEL OUTPUT. The isolation is deliberately modest - a
    subprocess, a timeout, a scratch cwd - and is NOT a security boundary
    against an adversarial model. It is adequate for benchmarking weights you
    chose to download. If that changes, put it in a container with
    --network none.

    THE SANDBOX NEEDS THE PACKAGES THE TESTS IMPORT. A correct reference
    solution scored 0 for us because numpy was missing from the environment,
    and it looked exactly like a model failure. Run `evalkit selftest` before
    trusting a zero.
    """
    program = extract_code(answer_code) + "\n\n" + test_code + "\n"
    with tempfile.TemporaryDirectory(prefix="evalkit-") as tmp:
        script = Path(tmp) / "case.py"
        script.write_text(program)
        try:
            proc = subprocess.run(
                [sys.executable, str(script)],
                capture_output=True, text=True, timeout=timeout_s, cwd=tmp,
                # HOME is redirected into the scratch dir: some libraries write
                # caches to ~ on import and fail noisily if it is unwritable.
                env={"PATH": "/usr/bin:/bin", "HOME": tmp,
                     "PYTHONDONTWRITEBYTECODE": "1"},
            )
        except subprocess.TimeoutExpired:
            return False, f"timed out after {timeout_s}s"
        if proc.returncode == 0:
            return True, "ok"
        detail = (proc.stderr or proc.stdout or "").strip().splitlines()
        return False, (detail[-1][:300] if detail else f"exit {proc.returncode}")


# Answer extraction for multiple choice. Ordered most to least explicit: a
# model that says "Answer: C" should not be scored on a stray "A" earlier in
# its reasoning.
_MCQ_PATTERNS = [
    r"(?:final\s+)?answer\s*(?:is)?\s*[:\-]?\s*\(?([A-J])\)?\b",
    r"\b(?:option|choice)\s*\(?([A-J])\)?\b",
    r"^\s*\(?([A-J])\)?\s*[.):]",
    r"\*\*\s*\(?([A-J])\)?\s*\*\*",
]


def extract_mcq(text: str, n_options: int) -> str | None:
    """Pull a letter answer out of a response, or None if there isn't one.

    None means UNSCORABLE, not wrong. A model that reasoned for 4,000 tokens
    and ran out of budget before stating a letter has not answered incorrectly;
    it has not answered. See report.py for why that distinction decides
    rankings.
    """
    t = (text or "").strip()
    if not t:
        return None
    valid = {chr(ord("A") + i) for i in range(n_options)}
    # Search the tail first: models state conclusions at the end.
    for chunk in (t[-400:], t):
        for pat in _MCQ_PATTERNS:
            for m in re.finditer(pat, chunk, re.I | re.M):
                letter = m.group(1).upper()
                if letter in valid:
                    return letter
    # Last resort: a bare letter on its own line.
    for line in reversed(t.splitlines()):
        s = line.strip().strip(".):(").upper()
        if len(s) == 1 and s in valid:
            return s
    return None


def score_case(case: dict, response: str) -> tuple[float | None, str]:
    """Score one case. Returns (score, detail); score None means unscorable."""
    kind = case.get("type")

    if kind == "mcq":
        opts = case.get("options") or []
        got = extract_mcq(response, len(opts))
        if got is None:
            return None, "no answer letter found"
        want = (case.get("answer") or "").strip().upper()
        return (1.0 if got == want else 0.0), f"answered {got}, expected {want}"

    if kind == "code":
        ok, why = run_python(response, case.get("test") or "")
        return (1.0 if ok else 0.0), why

    if kind == "ifeval":
        ok, why = verify_all(response, case.get("instructions") or [])
        return (1.0 if ok else 0.0), why

    if kind == "answered":
        ok, why = check_answered(response)
        return (1.0 if ok else 0.0), why

    return None, f"unknown case type {kind!r}"

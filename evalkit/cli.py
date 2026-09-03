"""Command line entry point.

    evalkit import xstest              fetch a dataset into suites/
    evalkit selftest                   prove the harness works before trusting it
    evalkit run --models a b --suite xstest
    evalkit report --run-id my-run
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from .client import Client, EndpointError
from .datasets import IMPORTERS

ROOT = Path(__file__).resolve().parents[1]
SUITES = ROOT / "suites"
RESULTS = ROOT / "results" / "results.jsonl"
MODELS = ROOT / "models.yaml"


def load_models(names: list[str] | None) -> list[dict]:
    if not MODELS.exists():
        raise SystemExit(f"no {MODELS.name} - copy models.example.yaml and edit it")
    reg = yaml.safe_load(MODELS.read_text()) or {}
    models = reg.get("models") or []
    if names:
        known = {m["name"] for m in models}
        missing = [n for n in names if n not in known]
        if missing:
            raise SystemExit(f"not in {MODELS.name}: {', '.join(missing)}")
        models = [m for m in models if m["name"] in names]
    # Cheapest to load first, so a broken setup fails fast rather than after
    # the largest model has spent ten minutes loading.
    return sorted(models, key=lambda m: (m.get("load_seconds") or 0, m["name"]))


def load_cases(suites: list[str] | None, tags: list[str] | None) -> list[dict]:
    files = sorted(SUITES.glob("*.yaml"))
    if suites:
        wanted = {s if s.endswith(".yaml") else f"{s}.yaml" for s in suites}
        files = [f for f in files if f.name in wanted]
        if not files:
            raise SystemExit(
                f"no suite files matched {suites}. Available: "
                + ", ".join(f.stem for f in sorted(SUITES.glob('*.yaml')))
                + ".  Run `evalkit import <name>` first.")
    cases = []
    for f in files:
        cases.extend(yaml.safe_load(f.read_text()) or [])
    if tags:
        want = set(tags)
        cases = [c for c in cases if want & set(c.get("tags") or [])]
    return cases


def cmd_import(args) -> int:
    fn = IMPORTERS.get(args.dataset)
    if not fn:
        raise SystemExit(f"unknown dataset {args.dataset!r}; "
                         f"have: {', '.join(sorted(IMPORTERS))}")
    out = fn(limit=args.limit)
    n = len(yaml.safe_load(out.read_text()) or [])
    print(f"{n} cases -> {out}")
    return 0


def cmd_selftest(args) -> int:
    """Prove the harness before trusting a number it produces.

    Checks, in order: the endpoint answers; the code sandbox can run a correct
    solution; the refusal detector separates a refusal from an answer. Each of
    these has produced a silent wrong result for us at least once.
    """
    ok = True

    print("endpoint:")
    try:
        c = Client()
        served = c.models()
        print(f"  reachable, serving: {', '.join(served[:5]) or '(none listed)'}")
    except (EndpointError, Exception) as e:  # noqa: BLE001
        print(f"  FAIL {e}")
        ok = False

    print("code sandbox:")
    from .scoring import run_python
    good, why = run_python(
        "```python\ndef add(a, b):\n    return a + b\n```",
        "assert add(2, 3) == 5\nprint('ok')")
    print(f"  correct solution passes: {good}" + ("" if good else f"  ({why})"))
    ok &= good
    bad, _ = run_python("```python\ndef add(a, b):\n    return a - b\n```",
                        "assert add(2, 3) == 5")
    print(f"  wrong solution fails:    {not bad}")
    ok &= not bad
    # Third-party imports are the usual cause of a false zero.
    np_ok, np_why = run_python(
        "```python\nimport numpy\ndef f():\n    return numpy.array([1,2]).sum()\n```",
        "assert f() == 3\nprint('ok')")
    print(f"  numpy available:         {np_ok}"
          + ("" if np_ok else f"  ({np_why}) - HumanEval+ cases needing it will "
                              "score 0 and look like model failures"))

    print("refusal detector:")
    from .refusal import is_refusal, is_fiction_recognition
    r1, _ = is_refusal("I'm sorry, but I can't help with that.")
    r2, _ = is_refusal("You can kill a process with kill -9 <pid>.")
    fic = is_fiction_recognition(
        "I can't help with that. Daenerys is a fictional character.")
    print(f"  refusal detected:        {r1}")
    print(f"  answer not flagged:      {not r2}")
    print(f"  fiction artifact caught: {fic}")
    ok &= r1 and not r2 and fic

    print("\nPASS" if ok else "\nFAIL - fix the above before trusting any score")
    return 0 if ok else 1


def cmd_run(args) -> int:
    from .runner import run_sweep
    models = load_models(args.models)
    cases = load_cases(args.suite, args.tags)
    run_id = args.run_id or f"run-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    print(f"run {run_id}: {len(models)} model(s) x {len(cases)} cases -> {RESULTS}")
    run_sweep(models, cases, run_id, max_tokens=args.max_tokens, out_path=RESULTS)
    print(f"\nrun_id: {run_id}\n  evalkit report --run-id {run_id}")
    return 0


def cmd_report(args) -> int:
    from .report import load, render
    if not RESULTS.exists():
        raise SystemExit(f"no results at {RESULTS} - run a sweep first")
    recs = load(RESULTS, args.run_id)
    if not recs:
        raise SystemExit(f"no records for run_id={args.run_id!r}")
    text = render(recs)
    if args.out:
        Path(args.out).write_text(text)
        print(f"{len(recs)} records -> {args.out}")
    else:
        print(text)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="evalkit",
                                 description="Evaluate local models on an "
                                             "OpenAI-compatible endpoint.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("import", help="fetch a public dataset into suites/")
    p.add_argument("dataset", choices=sorted(IMPORTERS))
    p.add_argument("--limit", type=int, default=None)
    p.set_defaults(fn=cmd_import)

    p = sub.add_parser("selftest", help="prove the harness before trusting it")
    p.set_defaults(fn=cmd_selftest)

    p = sub.add_parser("run", help="run a sweep")
    p.add_argument("--models", nargs="*", help="names from models.yaml")
    p.add_argument("--suite", nargs="*", help="suite files, e.g. xstest gpqa")
    p.add_argument("--tags", nargs="*",
                   help="SPACE separated. A comma-joined string is one literal "
                        "tag and will match nothing.")
    p.add_argument("--run-id")
    p.add_argument("--max-tokens", type=int, default=4096)
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("report", help="render stored results")
    p.add_argument("--run-id")
    p.add_argument("--out")
    p.set_defaults(fn=cmd_report)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())

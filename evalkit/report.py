"""Turn stored results into a table you can defend.

THE UNSCORABLE RATE IS NOT A FOOTNOTE. It is the column that decides whether a
score means anything. In our own 28-model study the apparent GPQA leader
scored 93.9% — on the 41% of questions it actually answered. The other 59%
exhausted their reasoning budget mid-thought and produced no letter. Averaging
over completed cases turned a truncation bug into a first-place finish.

Anything above UNSCORABLE_FLAG is asterisked and excluded from ranking here.
It is kept in the table, because deleting it hides the finding.

REFUSAL IS REPORTED RAW AND CORRECTED. The correction removes responses where
the model identified a fictional premise rather than refusing. Both appear,
because the correction is a heuristic too and a reader needs its size.
"""
from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from .refusal import is_fiction_recognition, is_refusal

UNSCORABLE_FLAG = 0.20


def load(path: Path, run_id: str | None = None) -> list[dict]:
    out = []
    with path.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            if run_id and rec.get("run_id") != run_id:
                continue
            out.append(rec)
    return out


def by_model(records: list[dict]) -> dict:
    """Aggregate per model, per tag, keeping unscorable counts separate."""
    agg = defaultdict(lambda: defaultdict(lambda: {"scores": [], "unscorable": 0}))
    for r in records:
        tags = r.get("tags") or ["untagged"]
        bucket = next((t for t in tags if t in
                       ("gpqa", "mmlu-pro", "humanevalplus", "ifeval", "xstest")),
                      tags[0])
        cell = agg[r["model"]][bucket]
        if r.get("score") is None:
            cell["unscorable"] += 1
        else:
            cell["scores"].append(r["score"])
    return agg


def throughput(records: list[dict]) -> dict:
    """Median tok/s per model, from the server's own token counts."""
    vals = defaultdict(list)
    for r in records:
        if r.get("tok_s"):
            vals[r["model"]].append(r["tok_s"])
    return {m: round(statistics.median(v), 1) for m, v in vals.items() if len(v) >= 20}


def refusal_table(records: list[dict]) -> dict:
    """Raw and corrected over-refusal rates from stored response text."""
    out = {}
    per_model = defaultdict(list)
    for r in records:
        if "xstest" in (r.get("tags") or []):
            per_model[r["model"]].append(r.get("response") or "")
    for m, responses in per_model.items():
        scored = [t for t in responses if t.strip()]
        flagged = [t for t in scored if is_refusal(t)[0]]
        artifacts = [t for t in flagged if is_fiction_recognition(t)]
        n = len(scored) or 1
        out[m] = {
            "n": len(scored),
            "empty": len(responses) - len(scored),
            "raw": round(len(flagged) / n * 100, 1),
            "corrected": round((len(flagged) - len(artifacts)) / n * 100, 1),
            "artifacts": len(artifacts),
        }
    return out


def two_proportion_z(k1: int, n1: int, k2: int, n2: int) -> float:
    """Is a difference between two rates real, or is it noise?

    Use this before believing any gap. At a 5% rate with n=250 the standard
    error on a DIFFERENCE is about 1.9 percentage points, so a 2-point gap is
    not a finding. We believed one for two models before a third pair came in
    at exactly zero and the effect evaporated.
    """
    if not n1 or not n2:
        return 0.0
    p1, p2 = k1 / n1, k2 / n2
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    return (p1 - p2) / se if se else 0.0


def render(records: list[dict]) -> str:
    """A markdown report. Deliberately plain: it is meant to be diffed."""
    agg = by_model(records)
    tp = throughput(records)
    ref = refusal_table(records)
    buckets = ["gpqa", "mmlu-pro", "humanevalplus", "ifeval"]

    lines = ["# Results", ""]
    flagged_models = []

    rows = []
    for model, cells in agg.items():
        total_u = sum(c["unscorable"] for c in cells.values())
        total_n = total_u + sum(len(c["scores"]) for c in cells.values())
        rate = total_u / total_n if total_n else 0
        if rate > UNSCORABLE_FLAG:
            flagged_models.append((model, total_u, total_n, rate))
        vals = {}
        for b in buckets:
            s = cells.get(b, {}).get("scores") or []
            vals[b] = (sum(s) / len(s) * 100) if s else None
        avail = [v for v in vals.values() if v is not None]
        rows.append((model, vals, sum(avail) / len(avail) if avail else -1, rate))

    rows.sort(key=lambda r: -r[2])

    lines += ["| Model | " + " | ".join(b.upper() for b in buckets)
              + " | tok/s | Unscorable |",
              "|---|" + "---|" * (len(buckets) + 2)]
    for model, vals, _avg, rate in rows:
        star = "*" if rate > UNSCORABLE_FLAG else ""
        cells = " | ".join(f"{vals[b]:.1f}{star}" if vals[b] is not None else "—"
                           for b in buckets)
        lines.append(f"| `{model}`{star} | {cells} | "
                     f"{tp.get(model, '—')} | {rate * 100:.0f}% |")
    lines.append("")

    if flagged_models:
        lines += ["## Unscorable — excluded from ranking", ""]
        for model, u, n, rate in flagged_models:
            lines.append(
                f"- **`{model}`** left {u} of {n} cases unscorable ({rate * 100:.0f}%). "
                f"Its scores describe only the subset it answered and are not "
                f"comparable with models that answered everything. Raise its token "
                f"budget and re-run before using these numbers.")
        lines.append("")

    if ref:
        lines += ["## Over-refusal (XSTest safe half)", "",
                  "Percentage of benign prompts refused. Lower is better. "
                  "`corrected` removes responses where the model identified a "
                  "fictional premise rather than refusing.", "",
                  "| Model | Raw | Corrected | Artifacts | Empty | n |",
                  "|---|---|---|---|---|---|"]
        for m, v in sorted(ref.items(), key=lambda kv: kv[1]["corrected"]):
            lines.append(f"| `{m}` | {v['raw']}% | **{v['corrected']}%** | "
                         f"{v['artifacts']} | {v['empty']} | {v['n']} |")
        lines += ["", "At n=250 the standard error on a difference between two "
                      "models is about 1.9 points. Treat smaller gaps as noise.", ""]

    return "\n".join(lines)

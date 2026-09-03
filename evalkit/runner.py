"""Run a sweep and record every case.

Two design choices carry most of the value here.

RESULTS ARE APPEND-ONLY JSONL, one record per case, holding the full response
text. Storing the response rather than only its score is what lets you rescore
later without re-running: every correction we made to our own scorers was
applied retroactively to data already on disk, and one of them changed a
published conclusion. A pipeline that stores only the number cannot do that.

MODELS ARE SWEPT IN LOAD ORDER, ONE AT A TIME. On a machine where a model swap
costs several minutes, interleaving models multiplies that cost by the number
of cases. This runs every case for one model before touching the next.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from .client import Client
from .scoring import score_case

RESULTS = Path(__file__).resolve().parents[1] / "results" / "results.jsonl"

# Abort a model after this many consecutive failures from the very first case.
# A model that fails its first three requests is misconfigured, not bad, and
# grinding through 250 cases to prove it wastes an hour.
EARLY_ABORT = 3


def run_sweep(models: list[dict], cases: list[dict], run_id: str,
              client: Client | None = None, max_tokens: int = 4096,
              out_path: Path | None = None, progress=print) -> dict:
    client = client or Client()
    out_path = out_path or RESULTS
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not cases:
        # A filter matching nothing is an error, not an empty sweep. Ours once
        # reported success having run zero of 1,047 cases, because --tags took
        # a comma-joined string as a single literal tag.
        raise SystemExit("no cases selected - check your --tags/--suite filter")
    if not models:
        raise SystemExit("no models selected")

    summary = {}
    for model in models:
        name = model["name"]
        served = model.get("served_name", name)
        budget = model.get("max_tokens", max_tokens)
        progress(f"\n{name}: {len(cases)} cases"
                 + (f" (first request loads the model, ~{model['load_seconds']}s)"
                    if model.get("load_seconds") else ""))
        started = time.monotonic()
        consecutive_failures = 0
        counts = {"scored": 0, "unscorable": 0, "errors": 0}

        for i, case in enumerate(cases, 1):
            rec = {
                "run_id": run_id, "model": name, "case_id": case["id"],
                "tags": case.get("tags", []),
                "at": datetime.now(timezone.utc).isoformat(),
            }
            try:
                got = client.complete(served, case["prompt"], max_tokens=budget)
                consecutive_failures = 0
            except Exception as e:  # noqa: BLE001
                consecutive_failures += 1
                counts["errors"] += 1
                rec.update({"error": f"{type(e).__name__}: {e}"[:300], "score": None})
                _append(out_path, rec)
                progress(f"    [{i}/{len(cases)}] {case['id']}: ERROR {rec['error'][:70]}")
                if consecutive_failures >= EARLY_ABORT and i <= EARLY_ABORT:
                    progress(f"    aborting {name}: {EARLY_ABORT} failures from the start")
                    break
                continue

            score, detail = score_case(case, got["text"])
            rec.update({
                "response": got["text"],
                "score": score,
                "detail": detail[:300],
                "seconds": got["seconds"],
                "completion_tokens": got["completion_tokens"],
                "tok_s": got["tok_s"],
                "finish_reason": got["finish_reason"],
            })
            _append(out_path, rec)
            counts["scored" if score is not None else "unscorable"] += 1
            progress(f"    [{i}/{len(cases)}] {case['id']}: "
                     f"{got['seconds']}s score={score}")

        mins = (time.monotonic() - started) / 60
        summary[name] = {**counts, "minutes": round(mins, 1)}
        progress(f"  {name}: done in {mins:.1f} min  "
                 f"scored={counts['scored']} unscorable={counts['unscorable']} "
                 f"errors={counts['errors']}")
        if counts["unscorable"] > 0.2 * max(len(cases), 1):
            # Flagged loudly here as well as in the report, because by the time
            # you read a leaderboard the number already looks authoritative.
            progress(f"  WARNING {name}: {counts['unscorable']} of {len(cases)} "
                     f"unscorable. Its score describes only the subset it "
                     f"answered. Raise its token budget and re-run.")
    return summary


def _append(path: Path, rec: dict) -> None:
    with path.open("a") as fh:
        fh.write(json.dumps(rec) + "\n")

"""Optional: read the fuzzy injury notes with a cheap Claude model.

Only runs when the ANTHROPIC_API_KEY secret is set. It sends just the
injury-report comments for players we care about and gets back structured
answers. Everything else in the pipeline is plain code.
"""
from __future__ import annotations

import json
import logging
import os

import requests

log = logging.getLogger(__name__)
MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-5-5")

PROMPT = """You read NBA injury-report notes. For each player below, decide from the
status and note only (do not guess beyond them):
- out: true if he will not play tonight (Out, or Doubtful, or the note says he is ruled out)
- minutes_limit: a number if the note mentions a minutes restriction or limit with a figure, else null
- limited: true if the note suggests reduced minutes without a figure (e.g. "on a minutes restriction", "easing back")
- bench: true if the note says he will come off the bench
Reply with only a JSON array: [{"player": "...", "out": bool, "minutes_limit": number|null, "limited": bool, "bench": bool}]

Players:
"""


def enabled() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def read_injuries(rows: list[dict]) -> dict[str, dict]:
    """rows: [{player, status, comment}] -> {player_lower: {out, minutes_limit, limited, bench}}"""
    if not enabled() or not rows:
        return {}
    body = PROMPT + "\n".join(f"- {r['player']} | {r['status']} | {r['comment']}" for r in rows)
    try:
        r = requests.post("https://api.anthropic.com/v1/messages", timeout=60, headers={
            "x-api-key": os.environ["ANTHROPIC_API_KEY"],
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }, json={"model": MODEL, "max_tokens": 2000, "messages": [{"role": "user", "content": body}]})
        r.raise_for_status()
        text = "".join(b.get("text", "") for b in r.json().get("content", []))
        start, end = text.find("["), text.rfind("]")
        parsed = json.loads(text[start:end + 1])
    except (requests.RequestException, ValueError, KeyError) as e:
        log.warning("AI injury read failed, using rules only: %s", e)
        return {}
    out = {str(p.get("player", "")).lower(): p for p in parsed if isinstance(p, dict)}
    usage = r.json().get("usage", {})
    log.info("Claude read %d injury notes (%s in / %s out tokens): %d out, %d limited, %d bench", len(rows),
             usage.get("input_tokens"), usage.get("output_tokens"), sum(1 for v in out.values() if v.get("out")),
             sum(1 for v in out.values() if v.get("limited") or v.get("minutes_limit")), sum(1 for v in out.values() if v.get("bench")))
    return out

"""Your manual overrides, read from overrides/nba.txt.

Format (one per line, # starts a comment):

    Walker Kessler: skip              -> never suggest this player
    Ja Morant: out                    -> treat as out (also changes teammates' splits)
    Luka Doncic: minutes 30           -> expected minutes tonight
    Austin Reaves 20+ points: odds 1.57   -> Sportsbet price for that leg
    Austin Reaves: overlap 0.7        -> roster continuity of his opponent (0-1)

Lines under a "date: YYYY-MM-DD" header only apply to that slate (Melbourne date).
Lines above any date header always apply. Put odds under a date header so they
don't carry over to another day.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path


@dataclass
class Overrides:
    skip: set[str] = field(default_factory=set)
    out: set[str] = field(default_factory=set)
    minutes: dict[str, float] = field(default_factory=dict)
    overlap: dict[str, float] = field(default_factory=dict)
    odds: dict[tuple[str, int, str], float] = field(default_factory=dict)  # (player, line, stat) -> price


STAT_WORDS = {"point": "points", "points": "points", "pts": "points",
              "rebound": "rebounds", "rebounds": "rebounds", "reb": "rebounds",
              "assist": "assists", "assists": "assists", "ast": "assists",
              "three": "threes", "threes": "threes", "3pm": "threes", "3s": "threes"}


def load(path: Path, slate: date) -> Overrides:
    ov = Overrides()
    if not path.exists():
        return ov
    active = True
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        m = re.match(r"^date:\s*(\d{4}-\d{2}-\d{2})$", line, re.I)
        if m:
            active = m.group(1) == slate.isoformat()
            continue
        if not active or ":" not in line:
            continue
        who, what = (s.strip() for s in line.rsplit(":", 1))
        what_l = what.lower()
        leg = re.match(r"^(.*?)\s+(\d+)\+\s*([a-z0-9]+)$", who, re.I)
        if what_l.startswith("odds") and leg:
            stat = STAT_WORDS.get(leg.group(3).lower())
            try:
                price = float(what_l.split()[1].lstrip("$"))
            except (IndexError, ValueError):
                continue
            if stat:
                ov.odds[(leg.group(1).strip().lower(), int(leg.group(2)), stat)] = price
        elif what_l == "skip":
            ov.skip.add(who.lower())
        elif what_l == "out":
            ov.out.add(who.lower())
        elif what_l.startswith("minutes"):
            try:
                ov.minutes[who.lower()] = float(what_l.split()[1])
            except (IndexError, ValueError):
                pass
        elif what_l.startswith("overlap"):
            try:
                ov.overlap[who.lower()] = max(0.0, min(1.0, float(what_l.split()[1])))
            except (IndexError, ValueError):
                pass
    return ov

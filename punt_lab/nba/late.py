"""Late checks: shortly before each game, re-check its injury news and prices.

    python -m punt_lab.nba.build --mode late

Runs every 30 minutes through the slate (see .github/workflows/nba.yml). Each game is checked
once, 15-75 minutes before its tip. The morning's multis are kept as they are (you may have
bet them); instead:
- legs whose player is now out or doubtful are marked, and any multi using them gets a
  warning with a suggested swap;
- the game is re-analysed with the fresh injury list, so teammates who gain from a late out
  get updated chances, and any that now look like value are added to the legs list;
- prices are refreshed from OddsPapi only when someone's status in that game changed since
  the morning (2 requests, at most twice a day), so the free plan still covers it.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta

from . import ai, espn, odds, oddspapi, overrides
from .build import ROOT, analyse, leg_row, load_json, save_json
from .model import multi_chance

log = logging.getLogger("punt_lab.nba.late")
WINDOW = (timedelta(minutes=15), timedelta(minutes=75))
MAX_REFRESHES = 2                       # OddsPapi price refreshes per day
REFRESH_BOOKS = ("neds.com.au", "pointsbet.com.au")
MAX_NEW_LEGS = 8                        # new legs added per check (value or 80%+)
BAD = ("out", "doubtful")


def restore_prices(legs: list[dict]) -> dict:
    prices = odds.empty()
    for l in legs:
        if l.get("market_line") is not None:
            prices["market"][(l["player"].lower(), l["stat"])] = (l["market_line"], l.get("market_p_over", 0.5),
                                                                 l.get("market_source") or "AU bookies")
        if l.get("books"):
            prices["books"][(l["player"].lower(), l["stat"], int(l["line"]))] = l["books"]
    return prices


def injury_snapshot(inj: dict, team_names: set[str]) -> dict[str, str]:
    """{player lower: status} for today's teams, saved by the morning run to spot late changes."""
    return {n: v["status"] for n, v in inj.items() if v.get("team") in team_names and v.get("status")}


def _target(multi: str) -> float | None:
    m = re.search(r"\$(\d+)", multi)
    return float(m.group(1)) if m else None


def _leg_price(l: dict) -> float:
    if l.get("best_price"):
        return l["best_price"]
    if l.get("fair_price"):
        return l["fair_price"] * 0.92
    return 0.92 / max(l.get("model_prob") or 0.5, 0.05)


def suggest_swap(multi: dict, bad_id: str, legs: list[dict], started: set[str]) -> dict | None:
    """The most likely leg not already in the multi, from a game yet to start, from a different
    player, that keeps the multi at or above its target price."""
    in_multi = [l for l in legs if l["leg_id"] in multi["leg_ids"]]
    players = {l["player"] for l in in_multi}
    rest = [l for l in in_multi if l["leg_id"] != bad_id]
    base = 1.0
    for l in rest:
        base *= _leg_price(l)
    target = _target(multi["multi"]) or 0
    options = [l for l in legs if l["leg_id"] not in multi["leg_ids"] and l["player"] not in players
               and l["game"] not in started and not l.get("late_out") and not l.get("questionable") and l.get("stat") != "win"
               and base * _leg_price(l) >= target]
    return max(options, key=lambda l: l["model_prob"], default=None)


def run(slate, now: datetime) -> bool:
    """Returns True if anything was written."""
    report = load_json("report.json", {})
    if report.get("slate_date") != slate.isoformat():
        log.info("No slate built for %s; nothing to check.", slate)
        return False
    games = espn.slate_games(slate)
    checked = set(report.get("late_checked", []))
    due = [g for g in games if g.get("state") in (None, "pre") and f"{g['away']} @ {g['home']}" not in checked
           and WINDOW[0] <= g["start"] - now <= WINDOW[1]]
    if not due:
        log.info("No game tips in the next 15-75 minutes that hasn't been checked.")
        return False
    labels = [f"{g['away']} @ {g['home']}" for g in due]
    started = {f"{g['away']} @ {g['home']}" for g in games if g.get("state") not in (None, "pre")}
    legs = load_json("legs.json", [])
    multis = load_json("multis.json", [])

    inj = espn.injuries()
    teams = {g["home_name"] for g in due} | {g["away_name"] for g in due}
    snap = report.get("inj_snapshot", {})
    now_snap = injury_snapshot(inj, teams)
    changed = sorted(n for n in set(snap) | set(now_snap)
                     if (snap.get(n) or "").lower() != (now_snap.get(n) or "").lower()
                     and any(w in (now_snap.get(n) or "").lower() for w in BAD + ("questionable", "day-to-day")))

    prices = restore_prices(legs)
    refreshed = ""
    if changed and oddspapi.enabled() and report.get("late_refreshes", 0) < MAX_REFRESHES:
        fresh = oddspapi.fetch([g["start"] for g in due], books=REFRESH_BOOKS)
        if fresh["books"]:
            prices = odds.finalise(odds.merge(prices, fresh))
            report["late_refreshes"] = report.get("late_refreshes", 0) + 1
            refreshed = "prices refreshed"

    ov = overrides.load(ROOT / "overrides" / "nba.txt", slate)
    # Claude reads the fresh notes for these games' players (minutes limits, bench roles), if a key is set
    rows = [{"player": n.title(), "status": v["status"], "comment": v["comment"]}
            for n, v in inj.items() if v.get("team") in teams and v.get("comment")]
    ai_inj = ai.read_injuries(rows[:60]) if rows else {}
    found = analyse(slate, [g for g in games if g.get("state") in (None, "pre")], ov, inj, ai_inj, prices, only=set(labels))
    fresh_rows = {r["leg_id"]: r for r in (leg_row(l, res) for l, res in found)}
    stamp = now.strftime("%-I:%M%p").lower()

    notes = []
    for i, l in enumerate(legs):
        if l["game"] not in labels or l.get("stat") == "win":
            continue
        status = (inj.get(l["player"].lower()) or {}).get("status", "")
        if status.lower() in BAD or l["player"].lower() in ov.out or (ai_inj.get(l["player"].lower()) or {}).get("out"):
            legs[i] = {**l, "late_out": status or "Out", "late_checked": stamp}
            notes.append(f"{l['player']} {status.lower() or 'out'}")
        elif l["leg_id"] in fresh_rows:
            new = fresh_rows.pop(l["leg_id"])
            legs[i] = {**new, "late_checked": stamp,
                       "late_was": l["model_prob"] if abs(new["model_prob"] - l["model_prob"]) >= 0.03 else None}
        else:
            legs[i] = {**l, "late_checked": stamp}
    # teammates who now look good: value at the best price, or 80%+
    extra = [r for r in fresh_rows.values() if not r.get("questionable") and not r.get("step_up")
             and ((r.get("best_price") and r["model_prob"] * r["best_price"] > 1.03) or r["model_prob"] >= 0.80)]
    extra.sort(key=lambda r: -(r["model_prob"] * (r.get("best_price") or 1)))
    known = {l["player"] + l["stat"] for l in legs}
    added = [{**r, "late_new": True, "late_checked": stamp} for r in extra if r["player"] + r["stat"] not in known][:MAX_NEW_LEGS]
    legs.extend(added)

    out_ids = {l["leg_id"] for l in legs if l.get("late_out")}
    for m in multis:
        bad = [i for i in m["leg_ids"] if i in out_ids]
        if not bad:
            continue
        msgs = []
        for b in bad:
            leg = next(l for l in legs if l["leg_id"] == b)
            swap = suggest_swap(m, b, legs, started)
            msgs.append(f"{leg['player']} is {leg['late_out'].lower()}. Bookies usually void that leg (the multi pays on the rest)"
                        + (f"; if you haven't bet yet, swap in {swap['player']} {swap['market']} ({swap['model_prob']:.0%})" if swap else ""))
        m["late_warning"] = "; ".join(msgs) + "."
        keep = [l for l in legs if l["leg_id"] in m["leg_ids"] and l["leg_id"] not in out_ids]
        m["est_hit_after_void"] = round(multi_chance([_FakeLeg(l) for l in keep], [l["model_prob"] for l in keep]), 4) if keep else None

    summary = ", ".join(labels) + ": " + ("; ".join(notes) if notes else "no new outs")
    if added:
        summary += f"; {len(added)} new leg{'s' if len(added) > 1 else ''} worth a look"
    if refreshed:
        summary += f"; {refreshed}"
    report["late_checked"] = sorted(checked | set(labels))
    report["late_log"] = (report.get("late_log") or []) + [f"{stamp} {summary}"]
    report["next_run"] = f"Late check {stamp} — {summary}"
    save_json("legs.json", legs)
    save_json("multis.json", multis)
    save_json("report.json", report)
    log.info("Late check: %s", summary)
    return True


class _FakeLeg:
    """Just enough of a Leg for multi_chance's same-game links."""
    def __init__(self, row: dict):
        self.game, self.team, self.stat = row["game"], row.get("team", ""), row["stat"]

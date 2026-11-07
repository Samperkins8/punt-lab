"""Daily pipeline: grade yesterday, analyse today's slate, pick multis, write the site data.

    python -m punt_lab.nba.build                 # morning run
    python -m punt_lab.nba.build --mode pretip   # re-check before the first tip
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from statistics import mean

from . import ai, espn, odds, overrides, pick
from .model import Leg, LegResult, evaluate, multi_chance, norm_cdf

log = logging.getLogger("punt_lab.nba")
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs" / "nba" / "data"

STATS = {"points": "pts", "rebounds": "reb", "assists": "ast", "threes": "threes"}
# Sportsbet-style milestone lines to consider for each stat
LINES = {"points": [10, 15, 20, 25, 30, 35, 40], "rebounds": [4, 6, 8, 10, 12, 14],
         "assists": [4, 6, 8, 10, 12], "threes": [2, 3, 4, 5]}
MIN_STAT_AVG = {"points": 12, "rebounds": 5, "assists": 4, "threes": 1.5}
ROTATION_MINUTES = 24
MIN_GAMES = 3
TARGET_LOW = 0.70   # lowest model chance worth listing as a leg
WIN_MARGIN = 0.95   # head-to-head prices carry a smaller bookie margin than player props
WIN_SD = 12.0       # points: spread -> win chance
STEP_UP_LOW = 0.55  # a player's next line up is kept for the $2+ multis if at least this likely
VENUE_GAMES = 8     # home/away split is halfway with this many matching games
PACE_WEIGHT = {"points": 0.5, "assists": 0.5, "threes": 0.5, "rebounds": 0.3}
B2B_HOURS = 30      # a game within this many hours of the last one is a back-to-back
ALL_TEAMS = {"ATL", "BOS", "BKN", "CHA", "CHI", "CLE", "DAL", "DEN", "DET", "GS", "HOU", "IND", "LAC", "LAL",
             "MEM", "MIA", "MIL", "MIN", "NO", "NY", "OKC", "ORL", "PHI", "PHX", "POR", "SAC", "SA", "TOR", "UTAH", "WSH"}
MAX_LEGS = 25


SUFFIXES = {"jr", "jr.", "sr", "sr.", "ii", "iii", "iv"}


def surname(name: str) -> str:
    parts = [p for p in name.split() if p.lower() not in SUFFIXES]
    return parts[-1] if parts else name


def bucket(pos: str | None) -> str:
    p = (pos or "").upper()
    if "C" in p:
        return "C"
    if "F" in p:
        return "F"
    return "G"


def season_end_year(d: date) -> int:
    return d.year + 1 if d.month >= 9 else d.year


def load_json(name: str, default):
    p = DATA / name
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except ValueError:
            pass
    return default


def save_json(name: str, data) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / name).write_text(json.dumps(data, indent=1, ensure_ascii=False, default=str), encoding="utf-8")


# ---------------------------------------------------------------- grading

def grade_previous(today: date) -> str:
    report = load_json("report.json", {})
    prev = report.get("slate_date")
    results = load_json("results.json", [])
    if not prev or prev >= today.isoformat() or any(r.get("slate_date") == prev for r in results):
        return ""
    legs = load_json("legs.json", [])
    multis = load_json("multis.json", [])
    boxes: dict[str, dict] = {}
    landed_by_leg: dict[str, bool | None] = {}
    for leg in legs:
        ev = leg.get("event_id")
        if ev and ev not in boxes:
            boxes[ev] = espn.box_score(ev)
        actual = None
        if leg["stat"] == "win":
            box = boxes.get(ev, {})
            if len(box) == 2 and leg["team"] in box:
                scores = {t: sum(r["pts"] or 0 for r in rows) for t, rows in box.items()}
                mine = scores.pop(leg["team"])
                actual = 1 if mine > max(scores.values()) else 0
        else:
          for rows in boxes.get(ev, {}).values():
            for row in rows:
                if row["id"] == str(leg.get("athlete_id")) and not row["dnp"]:
                    actual = row[STATS[leg["stat"]]]
        landed = None if actual is None else actual >= leg["line"]
        landed_by_leg[leg["leg_id"]] = landed
        results.append({
            "slate_date": prev, "leg_id": leg["leg_id"], "player": leg["player"], "game": leg["game"],
            "market": leg["market"], "line": leg["line"], "model_prob": leg["model_prob"], "raw_prob": leg.get("raw_prob"),
            "odds": leg.get("odds"), "fair_price": leg.get("fair_price"), "actual": actual, "landed": landed,
            "in_multis": ", ".join(m["multi"] for m in multis if leg["leg_id"] in m["leg_ids"]),
        })
    mres = load_json("multi_results.json", [])
    summary = []
    for m in multis:
        outs = [landed_by_leg.get(i) for i in m["leg_ids"]]
        landed = False if False in outs else (None if None in outs else True)
        mres.append({"slate_date": prev, "multi": m["multi"], "legs": m["legs_text"],
                     "est_hit": m["est_hit"], "landed": landed,
                     "price": m.get("known_price") if m.get("priced_legs", "").split(" of ")[0] == str(len(m["leg_ids"])) else m.get("est_price")})
        summary.append(f"{m['multi']} {'landed' if landed else 'void' if landed is None else 'missed'}")
    save_json("results.json", results)
    save_json("multi_results.json", mres)
    graded = [v for v in landed_by_leg.values() if v is not None]
    if not graded:
        return ""
    return f"Yesterday: {sum(graded)} of {len(graded)} legs landed; " + ", ".join(summary) + "."


# ---------------------------------------------------------------- analysis

def player_logs(team_id: str, team_abbr: str) -> tuple[list[str], dict[str, dict], dict[str, dict]]:
    """Returns (event ids in order, {athlete_id: {name, pos, games: {event_id: row}}}, {event_id: {date, home}})"""
    sched = espn.team_completed_games(team_id)
    events = [g["event_id"] for g in sched]
    meta = {g["event_id"]: g for g in sched}
    players: dict[str, dict] = {}
    for ev in events:
        rows = espn.box_score(ev).get(team_abbr, [])
        for row in rows:
            p = players.setdefault(row["id"], {"name": row["name"], "pos": row["pos"], "games": {}})
            if not row["dnp"]:
                p["games"][ev] = row
    return events, players, meta


def last_team(prior: dict) -> str | None:
    """Which team a player was on last season (most of it)."""
    teams = [g.get("team") for g in prior["games"] if g.get("team")]
    if teams:
        return max(set(teams), key=teams.count)
    opps = {g.get("opp") for g in prior["games"] if g.get("opp")}
    missing = ALL_TEAMS - opps
    return next(iter(missing)) if len(prior["games"]) >= 40 and len(missing) == 1 else None


def roster_overlap(td: dict, abbr: str, prior_year: int) -> float | None:
    """Share of a team's rotation minutes this season played by players who were on it last season."""
    total = same = 0.0
    known = 0
    for pid, p in td["players"].items():
        mins = [r["min"] for r in p["games"].values() if r["min"]]
        if len(mins) < MIN_GAMES or mean(mins) < 15:
            continue
        m = sum(mins)
        team = last_team(espn.last_season_log(pid, prior_year))
        if team is None:
            continue
        known += 1
        total += m
        same += m if team == abbr else 0
    return round(same / total, 2) if known >= 4 and total else None


def allowed_by_position(team_abbr: str, events: list[str]) -> dict[tuple[str, str], float]:
    """Per-game totals the team's opponents scored, by position bucket and stat."""
    totals = defaultdict(float)
    n = 0
    for ev in events:
        box = espn.box_score(ev)
        opp = [rows for abbr, rows in box.items() if abbr != team_abbr]
        if not opp:
            continue
        n += 1
        for row in opp[0]:
            if row["dnp"]:
                continue
            for stat, key in STATS.items():
                totals[(bucket(row["pos"]), stat)] += row[key] or 0
    return {k: v / n for k, v in totals.items()} if n else {}


def analyse(slate: date, games: list[dict], ov, inj: dict, ai_inj: dict, prices: dict) -> list[tuple[Leg, object]]:
    prior_year = season_end_year(slate) - 1
    team_data = {}
    for g in games:
        for side in ("home", "away"):
            abbr, tid = g[side], g[f"{side}_id"]
            if abbr and tid and abbr not in team_data:
                events, players, meta = player_logs(tid, abbr)
                team_data[abbr] = {"events": events, "players": players, "meta": meta}
                log.info("%s: %d games, %d players", abbr, len(events), len(players))

    allowed = {abbr: allowed_by_position(abbr, d["events"]) for abbr, d in team_data.items()}
    league = defaultdict(list)
    for a in allowed.values():
        for k, v in a.items():
            league[k].append(v)
    league_avg = {k: mean(v) for k, v in league.items() if v}

    totals = [g["total"] for g in games if g.get("total")]
    avg_total = mean(totals) if len(totals) >= 4 else 228.0
    overlap_cache: dict[str, float | None] = {}

    def is_out(name: str) -> bool:
        n = name.lower()
        if n in ov.out:
            return True
        a = ai_inj.get(n)
        if a and a.get("out"):
            return True
        st = (inj.get(n) or {}).get("status", "").lower()
        return st in ("out", "doubtful")

    cands = []
    for g in games:
        game_label = f"{g['away']} @ {g['home']}"
        spread = abs(g["spread"]) if g.get("spread") is not None else 0
        for side, opp_side in (("home", "away"), ("away", "home")):
            team, opp = g[side], g[opp_side]
            td = team_data.get(team)
            if not td:
                continue
            events = td["events"]
            avg_min = {pid: mean([r["min"] for r in p["games"].values() if r["min"]] or [0]) for pid, p in td["players"].items()}
            rotation = [pid for pid, p in td["players"].items()
                        if len(p["games"]) >= MIN_GAMES and avg_min[pid] >= ROTATION_MINUTES]
            # who is playing tonight
            playing = {pid: not is_out(td["players"][pid]["name"]) for pid in rotation}
            # roster continuity of the opponent: share of its rotation minutes by players who were there last season
            opp_td = team_data.get(opp)
            if opp not in overlap_cache:
                overlap_cache[opp] = roster_overlap(opp_td, opp, prior_year) if opp_td else None
            # back-to-back: this team played within the last ~30 hours
            last_dates = [td["meta"][ev]["date"] for ev in events if td["meta"].get(ev, {}).get("date")]
            b2b = False
            if last_dates:
                try:
                    last = datetime.fromisoformat(max(last_dates).replace("Z", "+00:00"))
                    b2b = 0 < (g["start"] - last).total_seconds() / 3600 < B2B_HOURS
                except ValueError:
                    pass
            at_home = side == "home"
            for pid in rotation:
                p = td["players"][pid]
                name = p["name"]
                if not playing[pid] or name.lower() in ov.skip:
                    continue
                played_events = [ev for ev in events if ev in p["games"]]
                if len(played_events) < MIN_GAMES:
                    continue
                prior = espn.last_season_log(pid, prior_year)
                h2h_games = [x for x in prior["games"] if x["opp"] == opp]
                note_bits = []
                inj_row = inj.get(name.lower())
                risky = False
                if inj_row and inj_row["status"]:
                    note_bits.append(f"{inj_row['status']}: {inj_row['comment'][:80]}".strip())
                    st_l, cm_l = inj_row["status"].lower(), inj_row["comment"].lower()
                    risky = ("questionable" in st_l or "day-to-day" in st_l or "questionable" in cm_l
                             or "game-time" in cm_l) and "probable" not in cm_l

                # minutes tonight
                proj_min = avg_min[pid]
                a = ai_inj.get(name.lower()) or {}
                if name.lower() in ov.minutes:
                    proj_min = ov.minutes[name.lower()]
                    note_bits.append(f"minutes set to {proj_min:g}")
                elif a.get("minutes_limit"):
                    proj_min = min(proj_min, float(a["minutes_limit"]))
                    note_bits.append(f"minutes limit {proj_min:g}")
                elif a.get("limited") or a.get("bench"):
                    proj_min *= 0.85
                    note_bits.append("expected to play reduced minutes")
                if b2b and name.lower() not in ov.minutes:
                    proj_min *= 0.95 if avg_min[pid] >= 30 else 0.97
                    note_bits.append("second night of a back-to-back")
                if spread >= 10:
                    proj_min *= 0.95
                    note_bits.append(f"blowout risk ({g['favourite']} {g['spread']:+.1f})")

                # situation: the teammate whose status tonight moves this player most
                for stat, key in STATS.items():
                    vals = [p["games"][ev][key] for ev in played_events if p["games"][ev][key] is not None]
                    if len(vals) < MIN_GAMES or mean(vals) < MIN_STAT_AVG[stat]:
                        continue
                    season_mean = mean(vals)
                    best_ctx = None
                    for tm in rotation:
                        if tm == pid:
                            continue
                        tm_games = td["players"][tm]["games"]
                        match = [p["games"][ev][key] for ev in played_events
                                 if (ev in tm_games) == playing[tm] and p["games"][ev][key] is not None]
                        if 2 <= len(match) < len(vals):
                            shift = (mean(match) - season_mean) * len(match) / (len(match) + 5)
                            if best_ctx is None or abs(shift) > abs(best_ctx[0]):
                                tm_full = td["players"][tm]["name"]
                                label = f"{tm_full} {'playing' if playing[tm] else 'out'} tonight"
                                detail = (f"{surname(name)} averaged {mean(match):.1f} in the {len(match)} games "
                                          f"{'with' if playing[tm] else 'without'} {surname(tm_full)} "
                                          f"({season_mean:.1f} across all {len(vals)}), so he's predicted "
                                          f"{abs(shift):.1f} {stat} "
                                          f"{'more' if shift > 0 else 'fewer'} tonight.")
                                best_ctx = (shift, label, match, detail)

                    venue_vals = [p["games"][ev][key] for ev in played_events
                                  if td["meta"].get(ev, {}).get("home") is at_home and p["games"][ev][key] is not None]
                    venue_shift = 0.0
                    if 2 <= len(venue_vals) < len(vals):
                        venue_shift = (mean(venue_vals) - season_mean) * len(venue_vals) / (len(venue_vals) + VENUE_GAMES)
                    pace_factor = 1.0
                    if g.get("total"):
                        pace_factor = max(0.95, min(1.05, 1 + PACE_WEIGHT[stat] * (g["total"] / avg_total - 1)))
                    if best_ctx and abs(best_ctx[0]) < 0.3:
                        best_ctx = None  # too small to matter: don't show it
                    prior_vals = [x[key] for x in prior["games"] if x.get(key) is not None]
                    prior_mean = mean(prior_vals) if prior_vals else season_mean
                    prior_games = len(prior_vals)
                    pos = bucket(p["pos"])
                    opp_allowed = allowed.get(opp, {}).get((pos, stat))
                    opp_factor = None
                    if opp_allowed and league_avg.get((pos, stat)):
                        opp_factor = round(max(0.85, min(1.15, opp_allowed / league_avg[(pos, stat)])), 2)
                    overlap = ov.overlap.get(name.lower(), overlap_cache.get(opp) if overlap_cache.get(opp) is not None else 0.5)
                    h2h = [x[key] for x in h2h_games if x.get(key) is not None] or None
                    mk = prices["market"].get((name.lower(), stat))

                    best = None
                    safe = None  # highest line still 80%+ likely and paying $1.15+, for the long multi
                    for line in LINES[stat]:
                        leg = Leg(
                            leg_id=f"{name.lower().replace(' ', '_').replace('.', '')}_{stat}{line}",
                            player=name, team=team, game=game_label, stat=stat, line=line, log=vals,
                            prior_mean=prior_mean, prior_games=prior_games,
                            odds=ov.odds.get((name.lower(), line, stat)) or prices["sportsbet"].get((name.lower(), stat, line)),
                            context_label=best_ctx[1] if best_ctx else None,
                            context_log=best_ctx[2] if best_ctx else None,
                            h2h_log=h2h, roster_overlap=overlap if h2h else None, opp_factor=opp_factor,
                            venue_shift=round(venue_shift, 2), pace_factor=round(pace_factor, 3),
                            avg_minutes=round(avg_min[pid], 1), proj_minutes=round(proj_min, 1),
                            market_line=mk[0] if mk else None, market_p_over=mk[1] if mk else None,
                            market_source=mk[2] if mk else None,
                            note="; ".join(note_bits),
                            extra={"event_id": g["event_id"], "athlete_id": pid, "risky": risky,
                                   "at_home": at_home, "venue_games": len(venue_vals), "b2b": b2b,
                                   "context_detail": best_ctx[3] if best_ctx else None},
                        )
                        res = evaluate(leg)
                        if res.model_prob < TARGET_LOW:
                            if best and not risky and res.model_prob >= STEP_UP_LOW:
                                leg.extra["step_up"] = True  # longer leg, only used to reach $2+ multis
                                cands.append((leg, res))
                            break  # higher lines only get less likely
                        if pick.long_ok(leg, res):
                            safe = (leg, res)
                        if pick.price_ok(leg, res):
                            best = (leg, res)  # keep the highest line that is still eligible
                    if best:
                        cands.append(best)
                    if safe and (not best or safe[0].line != best[0].line):
                        safe[0].extra["long_only"] = True
                        cands.append(safe)
    return cands


def match_winners(games: list[dict]) -> list[tuple[Leg, LegResult]]:
    """Favourite-to-win legs from the betting spread (free from ESPN)."""
    out = []
    for g in games:
        if g.get("spread") is None or not g.get("favourite"):
            continue
        fav = g["favourite"]
        if fav not in (g["home"], g["away"]):
            continue
        name = g["home_name"] if fav == g["home"] else g["away_name"]
        p = norm_cdf(abs(g["spread"]) / WIN_SD)
        leg = Leg(leg_id=f"{fav.lower()}_win", player=name, team=fav, game=f"{g['away']} @ {g['home']}",
                  stat="win", line=1, log=[], prior_mean=0, prior_games=0, market=f"{name} to win",
                  note=f"{abs(g['spread']):g}-point favourite",
                  extra={"event_id": g["event_id"], "margin": WIN_MARGIN, "long_only": True,
                         "spread": abs(g["spread"])})
        res = LegResult(hit_rate=0.0, context_shift=0.0, matchup=1.0, minutes_factor=1.0, proj_mean=0.0,
                        proj_sd=0.0, market_mean=None, fair_price=1 / p, model_prob=p, implied=None, edge=None)
        if leg.odds is None and pick.long_ok(leg, res):
            out.append((leg, res))
    return out


def trim(cands):
    cands.sort(key=lambda c: -c[1].model_prob)
    keep, games = [], set()
    for c in cands:  # at least one per game first
        if c[0].game not in games:
            keep.append(c)
            games.add(c[0].game)
    for c in cands:
        if c not in keep and len(keep) < MAX_LEGS:
            keep.append(c)
    keep.sort(key=lambda c: -c[1].model_prob)
    return keep


# ---------------------------------------------------------------- output

def explain(leg: Leg, res) -> str:
    """One plain sentence or three on why this leg's prediction is what it is."""
    if leg.stat == "win":
        return (f"{leg.player} are {leg.extra['spread']:g}-point favourites. The betting line makes that "
                f"about a {res.model_prob:.0%} chance to win, fair price ${res.fair_price:.2f}.")
    unit = leg.stat
    season = mean(leg.log)
    parts = [f"Predicted {res.proj_mean:.1f} {unit} tonight, needs {int(leg.line)}+."]
    base = f"This season {season:.1f} a game over {len(leg.log)} games"
    if leg.prior_games:
        base += f"; last season {leg.prior_mean:.1f} over {leg.prior_games}"
    parts.append(base + ".")
    if leg.extra.get("context_detail"):
        parts.append(leg.extra["context_detail"])
    if abs(leg.venue_shift) >= 0.3:
        where = "at home" if leg.extra.get("at_home") else "on the road"
        parts.append(f"Plays better {where}" if leg.venue_shift > 0 else f"Plays a bit worse {where}")
        parts[-1] += f": {abs(leg.venue_shift):.1f} {unit} {'more' if leg.venue_shift > 0 else 'fewer'} predicted."
    if abs(leg.pace_factor - 1) >= 0.01:
        parts.append(f"{'Faster' if leg.pace_factor > 1 else 'Slower'} game expected ({leg.pace_factor - 1:+.0%} for pace).")
    if abs(res.matchup - 1) >= 0.02:
        pct = round((res.matchup - 1) * 100)
        who = leg.game.replace(leg.team, "").replace("@", "").strip()
        parts.append(f"{who} {'gives up more' if pct > 0 else 'gives up less'} to his position than average ({pct:+d}%).")
    if abs(res.minutes_factor - 1) >= 0.02:
        parts.append(f"Expected minutes {leg.proj_minutes:g} vs his usual {leg.avg_minutes:g}.")
    if leg.market_line is not None:
        parts.append(f"US betting line: {leg.market_line:g}.")
    return " ".join(parts)


def leg_row(leg: Leg, res) -> dict:
    return {
        "why": explain(leg, res),
        "venue_shift": leg.venue_shift, "pace_factor": leg.pace_factor, "roster_overlap": leg.roster_overlap,
        "back_to_back": leg.extra.get("b2b", False),
        "leg_id": leg.leg_id, "player": leg.player, "team": leg.team, "game": leg.game,
        "market": leg.market, "stat": leg.stat, "line": leg.line,
        "hits": sum(1 for x in leg.log if x >= leg.line), "games": len(leg.log),
        "log": ", ".join(f"{round(x, 1):g}" for x in leg.log), "odds": leg.odds,
        "prior_mean": round(leg.prior_mean, 1), "prior_games": leg.prior_games,
        "context_label": leg.context_label, "context_detail": leg.extra.get("context_detail"), "matchup": round(res.matchup, 3),
        "context_shift": round(res.context_shift, 2), "minutes_factor": round(res.minutes_factor, 3),
        "avg_minutes": leg.avg_minutes, "proj_minutes": leg.proj_minutes,
        "proj_mean": round(res.proj_mean, 1), "model_prob": round(res.model_prob, 4),
        "raw_prob": round(res.raw_prob, 4) if res.raw_prob is not None else None,
        "fair_price": round(res.fair_price, 2) if res.fair_price else None,
        "implied": round(res.implied, 4) if res.implied else None,
        "edge": round(res.edge, 4) if res.edge is not None else None,
        "hit_rate": round(res.hit_rate, 4), "market_line": leg.market_line, "market_p_over": leg.market_p_over,
        "questionable": leg.extra.get("risky", False), "step_up": leg.extra.get("step_up", False), "note": leg.note, "event_id": leg.extra.get("event_id"), "athlete_id": leg.extra.get("athlete_id"),
    }


def multi_rows(cands, chosen) -> list[dict]:
    rows = []
    order = ("4-leg $2+", "3-leg $2+", "4-leg $3+", "4-leg $4+", "Best value", "Safest", "3-leg value", "One per game", "Mid-range $8+", "Long multi 8+", "Long shot $50")
    rank = {n: i + 1 for i, n in enumerate(order)}
    for name in order:
        ids = chosen.get(name)
        if not ids:
            continue
        legs = [cands[i][0] for i in ids]
        probs = [cands[i][1].model_prob for i in ids]
        p = multi_chance(legs, probs)
        known = [l.odds for l in legs if l.odds]
        fair = [cands[i][1].fair_price for i in ids]
        price = 1.0
        for k in known:
            price *= k
        rows.append({
            "multi": name, "rank": rank[name], "leg_ids": [l.leg_id for l in legs],
            "legs_text": "; ".join(f"{l.player} {l.market}" for l in legs),
            "est_hit": round(p, 4), "break_even": round(1 / p, 2) if p else None,
            "priced_legs": f"{len(known)} of {len(legs)}",
            "known_price": round(price, 2) if known else None,
            "fair_multi": round(_prod(fair), 2) if all(fair) else None,
            "est_price": round(_prod([pick._assumed_price(*cands[i]) for i in ids]), 2),
        })
    return rows


def _prod(xs):
    out = 1.0
    for x in xs:
        out *= x
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["daily", "pretip"], default="daily")
    ap.add_argument("--date", help="slate date YYYY-MM-DD (Melbourne); default today")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    now = datetime.now(espn.MELB)
    # Evening runs (after 4pm Melbourne) look ahead to the next morning's games.
    default_slate = now.date() + timedelta(days=1) if now.hour >= 16 else now.date()
    slate = date.fromisoformat(args.date) if args.date else default_slate
    graded = grade_previous(slate) if args.mode == "daily" else ""

    games = espn.slate_games(slate)
    if args.mode == "pretip":
        prev_multis = load_json("multis.json", [])
        if load_json("report.json", {}).get("slate_date") != slate.isoformat():
            log.info("No slate built for today; nothing to re-check.")
            return
    if not games:
        save_json("report.json", {**load_json("report.json", {}), "status": f"No games on {slate:%a %-d %b}. {graded}".strip()})
        log.info("No games today.")
        return
    upcoming = [g for g in games if g.get("state") in (None, "pre")]

    ov = overrides.load(ROOT / "overrides" / "nba.txt", slate)
    inj = espn.injuries()
    prices = odds.fetch(upcoming) if args.mode == "daily" else {"market": {}, "sportsbet": {}}
    if args.mode == "pretip":  # keep the morning's prices
        for l in load_json("legs.json", []):
            if l.get("market_line") is not None:
                prices["market"][(l["player"].lower(), l["stat"])] = (l["market_line"], l.get("market_p_over", 0.5), "The Odds API")
            if l.get("odds"):
                prices["sportsbet"][(l["player"].lower(), l["stat"], int(l["line"]))] = l["odds"]

    # optional AI read of injury notes for players on today's teams
    teams_today = {g["home"] for g in upcoming} | {g["away"] for g in upcoming}
    ai_rows = [{"player": n.title(), "status": v["status"], "comment": v["comment"]}
               for n, v in inj.items() if v.get("comment")]
    ai_inj = ai.read_injuries(ai_rows[:120]) if teams_today else {}

    found = analyse(slate, upcoming, ov, inj, ai_inj, prices)
    extra_kinds = ("step_up", "long_only")
    main_legs = trim([c for c in found if not any(c[0].extra.get(k) for k in extra_kinds)])
    keep_players = {c[0].player for c in main_legs}
    step_ups = [c for c in found if c[0].extra.get("step_up") and c[0].player in keep_players]
    safe_legs = [c for c in found if c[0].extra.get("long_only")] + match_winners(upcoming)
    cands = main_legs + step_ups + safe_legs
    chosen = pick.choose(cands)
    used = {i for k, v in chosen.items() if not k.startswith("_") for i in v}
    kept = [i for i in range(len(cands)) if i < len(main_legs) or i in used]
    remap = {old: new for new, old in enumerate(kept)}
    cands = [cands[i] for i in kept]
    chosen = {k: (v if k.startswith("_") else [remap[i] for i in v]) for k, v in chosen.items()}
    legs_out = [leg_row(l, r) for l, r in cands]
    multis_out = multi_rows(cands, chosen)

    changes = ""
    if args.mode == "pretip":
        before = {m["multi"]: set(m["leg_ids"]) for m in prev_multis}
        after = {m["multi"]: set(m["leg_ids"]) for m in multis_out}
        changed = [k for k in after if before.get(k) != after[k]]
        changes = f"{len(changed)} multi(s) changed: {', '.join(changed)}" if changed else "no changes"

    slate_out = []
    for g in games:
        fav_win = None
        if g.get("spread") is not None:
            from .model import norm_cdf
            fav_win = round(norm_cdf(abs(g["spread"]) / 12), 3)
        slate_out.append({
            "game": f"{g['away']} @ {g['home']}", "tip": g["start"].strftime("%-I:%M%p").lower(),
            "favourite": g.get("favourite"), "spread": g.get("spread"), "total": g.get("total"),
            "fav_win_prob": fav_win, "status": "Analysed" if g in upcoming else "Started",
        })

    top = next((m for m in multis_out if m["multi"] == "3-leg $2+"), None) or (multis_out[0] if multis_out else None)
    value_leg = max(cands, key=lambda c: c[1].edge if c[1].edge is not None else -1, default=None)
    headline = ""
    if top:
        headline = f"{top['multi']}: {top['legs_text']} — about {top['est_hit']:.0%} to land, est. price ${top['est_price']:.2f}, worth it at ${top['break_even']:.2f}+."
    if value_leg and value_leg[1].edge and value_leg[1].edge > 0:
        headline += f" Best-priced leg: {value_leg[0].player} {value_leg[0].market} at ${value_leg[0].odds:.2f}."
    if chosen.get("_notes"):
        headline += " Note: " + "; ".join(chosen["_notes"]) + "."
    slate_teams = {g["home_name"] for g in games} | {g["away_name"] for g in games}
    outs = [f"{n.title()} ({v['status']})" for n, v in inj.items()
            if v["status"].lower() in ("out", "doubtful") and v.get("team") in slate_teams]
    stamp = now.strftime("%-I:%M%p").lower()
    report = {
        "slate_date": slate.isoformat(), "headline": headline,
        "injuries": ", ".join(outs[:12]) + (" and others" if len(outs) > 12 else ""),
        "graded": graded,
        "next_run": f"Re-checked {stamp}: {changes}" if args.mode == "pretip" else f"Built {stamp}; re-checked before the first tip",
        "sources": {"stats": "ESPN", "odds": "The Odds API" if odds.enabled() else None,
                    "injury_reading": "Claude" if ai.enabled() else "rules"},
        "updated_at": now.isoformat(),
    }
    save_json("legs.json", legs_out)
    save_json("multis.json", multis_out)
    save_json("slate.json", slate_out)
    save_json("report.json", report)
    log.info("Wrote %d legs, %d multis. %s", len(legs_out), len(multis_out), changes or graded)


if __name__ == "__main__":
    main()

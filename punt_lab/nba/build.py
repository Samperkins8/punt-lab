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

from . import ai, espn, model, odds, oddspapi, overrides, pick
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
STEP_UP_LOW = 0.40  # a player's next line up is kept if at least this likely ($2+ multis only use those 55%+; see pick.py)
VENUE_GAMES = 8     # home/away split is halfway with this many matching games
STARTER_RATE = 0.6  # a player who started at least this share of his games is a usual starter
ROLE_GAMES = 2      # his role has changed if his last this-many games all went the other way
ROLE_MIN_GAMES = 5  # ...and he has at least this many games to judge his usual role from
PACE_WEIGHT = {"points": 0.5, "assists": 0.5, "threes": 0.5, "rebounds": 0.3}
B2B_HOURS = 30      # a game within this many hours of the last one is a back-to-back
ALL_TEAMS = {"ATL", "BOS", "BKN", "CHA", "CHI", "CLE", "DAL", "DEN", "DET", "GS", "HOU", "IND", "LAC", "LAL",
             "MEM", "MIA", "MIL", "MIN", "NO", "NY", "OKC", "ORL", "PHI", "PHX", "POR", "SAC", "SA", "TOR", "UTAH", "WSH"}
MAX_LEGS = 25
VALUE_MARGIN = 0.05   # a multi is "value" when its app price beats break-even by this much: the chances are estimates
SKIP_MARGIN = -0.03   # ...and a "skip" when it falls this far short; in between is marginal


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


# ---------------------------------------------------------------- the morning's prices

def save_prices(slate: date, prices: dict) -> None:
    """Keep every price and bookie line the morning run fetched, so the re-checks later in the day
    still know them for legs that weren't listed in the morning."""
    save_json("prices.json", {
        "slate_date": slate.isoformat(),
        "books": [[p, stat, line, bk] for (p, stat, line), bk in prices["books"].items() if bk],
        "market": [[p, stat, *v] for (p, stat), v in prices["market"].items() if isinstance(v, tuple)],
    })


def load_prices(slate: date) -> dict | None:
    """The morning's prices for this slate, or None if there are none (then the legs' own prices are used)."""
    data = load_json("prices.json", {})
    if data.get("slate_date") != slate.isoformat() or not (data.get("books") or data.get("market")):
        return None
    prices = odds.empty()
    for p, stat, line, bk in data.get("books", []):
        prices["books"][(p, stat, int(line))] = bk
    for row in data.get("market", []):
        prices["market"][(row[0], row[1])] = tuple(row[2:])
    return prices


# ---------------------------------------------------------------- grading

def leg_actual(leg: dict, boxes: dict) -> float | None:
    """The leg's stat from the box score (1/0 for a win leg), or None if he didn't play or there's no score."""
    ev = leg.get("event_id")
    if ev and ev not in boxes:
        boxes[ev] = espn.box_score(ev)
    if leg["stat"] == "win":
        box = boxes.get(ev, {})
        if len(box) == 2 and leg["team"] in box:
            scores = {t: sum(r["pts"] or 0 for r in rows) for t, rows in box.items()}
            mine = scores.pop(leg["team"])
            return 1 if mine > max(scores.values()) else 0
        return None
    for rows in boxes.get(ev, {}).values():
        for row in rows:
            if row["id"] == str(leg.get("athlete_id")) and not row["dnp"]:
                return row[STATS[leg["stat"]]]
    return None


def grade_live(now: datetime) -> bool:
    """Grades today's legs as each game finishes, so the site can show a multi as landed or missed
    straight after its last game, not the next morning. Uses ESPN's free box scores, no odds credits.
    Returns True if anything was written."""
    report = load_json("report.json", {})
    slate = report.get("slate_date")
    if not slate:
        return False
    done = set(report.get("final_games", []))
    games = espn.slate_games(date.fromisoformat(slate))
    finished = {f"{g['away']} @ {g['home']}" for g in games if g.get("state") == "post"} - done
    if not finished:
        return False
    legs = load_json("legs.json", [])
    boxes: dict = {}
    for leg in legs:
        if leg["game"] in finished:
            actual = leg_actual(leg, boxes)
            leg["actual"] = actual
            leg["landed"] = None if actual is None else actual >= leg["line"]
    done |= finished
    by_id = {l["leg_id"]: l for l in legs}
    multis = load_json("multis.json", [])
    for m in multis:
        ms = [by_id[i] for i in m["leg_ids"] if i in by_id]
        if any(l.get("landed") is False for l in ms):
            m["landed"] = False
        elif ms and all(l["game"] in done for l in ms) and any(l.get("landed") for l in ms):
            m["landed"] = True  # a leg whose player didn't play is void; the rest got up
        else:
            m["landed"] = None
    report["final_games"] = sorted(done)
    save_json("legs.json", legs)
    save_json("multis.json", multis)
    save_json("report.json", report)
    log.info("Graded finished games: %s", ", ".join(sorted(finished)))
    return True


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
        actual = leg_actual(leg, boxes)
        landed = None if actual is None else actual >= leg["line"]
        landed_by_leg[leg["leg_id"]] = landed
        results.append({
            "slate_date": prev, "leg_id": leg["leg_id"], "player": leg["player"], "game": leg["game"],
            "market": leg["market"], "stat": leg.get("stat"), "line": leg["line"], "model_prob": leg["model_prob"], "raw_prob": leg.get("raw_prob"), "best_price": leg.get("best_price"),
            "odds": leg.get("odds"), "fair_price": leg.get("fair_price"), "actual": actual, "landed": landed,
            "in_multis": ", ".join(m["multi"] for m in multis if leg["leg_id"] in m["leg_ids"]),
        })
    mres = load_json("multi_results.json", [])
    summary = []
    for m in multis:
        outs = [landed_by_leg.get(i) for i in m["leg_ids"]]
        landed = False if False in outs else (None if None in outs else True)
        mres.append({"slate_date": prev, "multi": m["multi"], "legs": m["legs_text"],
                     "est_hit": m["est_hit"], "landed": landed, "verdict": m.get("verdict"),
                     "price": m.get("best_book_price") or m.get("known_price") if m.get("priced_legs", "").split(" of ")[0] == str(len(m["leg_ids"])) else m.get("est_price")})
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
        box = espn.box_score(ev)
        rows = box.get(team_abbr, [])
        others = [r for abbr, rs in box.items() if abbr != team_abbr for r in rs]
        if rows and others:  # the box score has every player, so the points add up to the final score
            meta[ev]["margin"] = sum(r["pts"] or 0 for r in rows) - sum(r["pts"] or 0 for r in others)
        for row in rows:
            p = players.setdefault(row["id"], {"name": row["name"], "pos": row["pos"], "games": {}})
            if not row["dnp"]:
                p["games"][ev] = row
    return events, players, meta


def role_change(starts: list[bool], minutes: list[float]) -> tuple[str, float] | None:
    """A usual starter who came off the bench in his last couple of games (or a bench player who started
    them) plays his new role's minutes tonight, not his season average. Returns (note, minutes) or None."""
    if len(starts) < ROLE_MIN_GAMES or len(minutes) != len(starts):
        return None
    recent, recent_min = starts[-ROLE_GAMES:], [m for m in minutes[-ROLE_GAMES:] if m]
    earlier = starts[:-ROLE_GAMES]
    if not recent_min or not earlier:
        return None
    rate = sum(earlier) / len(earlier)
    if rate >= STARTER_RATE and not any(recent):
        return f"came off the bench in his last {ROLE_GAMES} games", mean(recent_min)
    if rate <= 1 - STARTER_RATE and all(recent):
        return f"started his last {ROLE_GAMES} games", mean(recent_min)
    return None


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


def analyse(slate: date, games: list[dict], ov, inj: dict, ai_inj: dict, prices: dict,
            only: set[str] | None = None) -> list[tuple[Leg, object]]:
    """Legs for every game in `games` (or just the games labelled in `only`; the rest still
    count towards league averages for defence and pace)."""
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
        if only is not None and f"{g['away']} @ {g['home']}" not in only:
            continue
        game_label = f"{g['away']} @ {g['home']}"
        spread = abs(g["spread"]) if g.get("spread") is not None else 0
        fav = g.get("favourite") if g.get("favourite") in (g["home"], g["away"]) else None
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
            # tonight's line against this team's usual result, with home court taken out of the line
            line_shift = None
            margins = [td["meta"][ev]["margin"] for ev in events if td["meta"].get(ev, {}).get("margin") is not None]
            if fav is not None and len(margins) >= MIN_GAMES:
                signed = spread if team == fav else -spread
                line_shift = round(signed - (model.HOME_EDGE if at_home else -model.HOME_EDGE) - model.rating(margins), 1)
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
                starts = [bool(p["games"][ev].get("starter")) for ev in played_events]
                role = role_change(starts, [p["games"][ev]["min"] or 0 for ev in played_events])
                if role and name.lower() not in ov.minutes:
                    proj_min = role[1]
                    note_bits.append(role[0])
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
                    proj_min *= model.B2B_CUT_STAR if avg_min[pid] >= 30 else model.B2B_CUT
                    note_bits.append("second night of a back-to-back")
                underdog = fav is not None and team != fav
                cut = model.blowout_cut((-spread if underdog else spread) if fav is not None else None)
                if cut < 1:
                    proj_min *= cut
                    note_bits.append(f"blowout risk ({g['favourite']} {-spread:g}"
                                     f"{', he is on the underdog' if underdog else ''})")

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
                    side_factor = model.side_factor(stat, line_shift, g.get("total")) if line_shift is not None else 1.0
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
                            odds=ov.odds.get((name.lower(), line, stat)) or odds.typical(odds.allowed(prices["books"].get((name.lower(), stat, line)))),
                            context_label=best_ctx[1] if best_ctx else None,
                            context_log=best_ctx[2] if best_ctx else None,
                            h2h_log=h2h, roster_overlap=overlap if h2h else None, opp_factor=opp_factor,
                            venue_shift=round(venue_shift, 2), pace_factor=round(pace_factor, 3),
                            side_factor=round(side_factor, 3),
                            avg_minutes=round(avg_min[pid], 1), proj_minutes=round(proj_min, 1),
                            market_line=mk[0] if mk else None, market_p_over=mk[1] if mk else None,
                            market_source=mk[2] if mk else None,
                            note="; ".join(note_bits),
                            extra={"event_id": g["event_id"], "athlete_id": pid, "risky": risky,
                                   "books": odds.allowed(prices["books"].get((name.lower(), stat, line))),
                                   "at_home": at_home, "venue_games": len(venue_vals), "b2b": b2b,
                                   "underdog": underdog, "spread": spread, "line_shift": line_shift,
                                   "usual_starter": sum(starts) >= STARTER_RATE * len(starts),
                                   "context_detail": best_ctx[3] if best_ctx else None},
                        )
                        res = evaluate(leg)
                        if res.model_prob < TARGET_LOW:
                            if best and not risky and res.model_prob >= STEP_UP_LOW:
                                leg.extra["step_up"] = True  # longer leg, only used to reach a target price
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

def price_source(prices: dict) -> str | None:
    bits = []
    if oddspapi.enabled():
        bits.append("OddsPapi (PointsBet, Neds, Unibet)")
    if prices.get("priced"):
        bits.append(f"The Odds API ({len(prices['priced'])} games"
                    + (f", {prices['credits_left']} credits left" if prices.get("credits_left") is not None else "") + ")")
    elif odds.enabled():
        bits.append("The Odds API")
    return " + ".join(bits) or None


def select(found, upcoming):
    """Trim the candidate legs, pick the multis, and keep only the legs worth listing."""
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
    return cands, chosen


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
    if abs(leg.side_factor - 1) >= 0.01:
        shift = leg.extra.get("line_shift") or 0
        parts.append(f"Tonight's line is {abs(shift):.0f} points {'tougher' if shift < 0 else 'easier'} than his team's "
                     f"usual ({'underdogs' if leg.extra.get('underdog') else 'favourites'} by {leg.extra.get('spread', 0):g}), "
                     f"so they're expected to score {'less' if leg.side_factor < 1 else 'more'} ({leg.side_factor - 1:+.0%}).")
    if abs(res.matchup - 1) >= 0.02:
        pct = round((res.matchup - 1) * 100)
        who = leg.game.replace(leg.team, "").replace("@", "").strip()
        parts.append(f"{who} {'gives up more' if pct > 0 else 'gives up less'} to his position than average ({pct:+d}%).")
    if abs(res.minutes_factor - 1) >= 0.02:
        parts.append(f"Expected minutes {leg.proj_minutes:g} vs his usual {leg.avg_minutes:g}.")
    if leg.market_line is not None:
        parts.append(f"Bookies' line: {leg.market_line:g}.")
    return " ".join(parts)


def leg_row(leg: Leg, res) -> dict:
    return {
        "why": explain(leg, res),
        "venue_shift": leg.venue_shift, "pace_factor": leg.pace_factor, "side_factor": leg.side_factor,
        "underdog": leg.extra.get("underdog", False), "roster_overlap": leg.roster_overlap,
        "usual_starter": leg.extra.get("usual_starter"),
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
        "books": leg.extra.get("books") or None, "best_book": odds.best(leg.extra.get("books"))[0],
        "best_price": odds.best(leg.extra.get("books"))[1], "market_source": leg.market_source,
        "questionable": leg.extra.get("risky", False), "step_up": leg.extra.get("step_up", False), "note": leg.note, "event_id": leg.extra.get("event_id"), "athlete_id": leg.extra.get("athlete_id"),
    }


def pool_rows(found, winners) -> list[dict]:
    """Every leg the model rates, not just the listed ones: the site's promo builder searches these
    (a same-game promo needs four or more legs from one game, more than the main list keeps)."""
    rows = []
    for leg, res in list(found) + list(winners):
        books = leg.extra.get("books") or None
        rows.append({
            "leg_id": leg.leg_id, "player": leg.player, "team": leg.team, "game": leg.game, "stat": leg.stat,
            "line": leg.line, "market": leg.market, "model_prob": round(res.model_prob, 4),
            "books": books, "best_price": odds.best(books)[1] or leg.odds,
            "fair_price": round(res.fair_price, 2) if res.fair_price else None,
            "questionable": leg.extra.get("risky", False), "hits": sum(1 for x in leg.log if x >= leg.line),
            "games": len(leg.log),
        })
    rows.sort(key=lambda r: -r["model_prob"])
    return rows


def multi_rows(cands, chosen) -> list[dict]:
    rows = []
    order = ("4-leg $2+", "3-leg $2+", "2-leg $2+", "4-leg $3+", "4-leg $4+", "Best value", "Safest", "3-leg value", "One per game", "Mid-range $8+", "Long multi 8+", "Long shot $50")
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
            "est_price": round(pick.combo_price(cands, ids), 2),
            **_multi_books(legs),
        })
        rows[-1].update(verdict(rows[-1]))
    return rows


def verdict(m: dict) -> dict:
    """Value / marginal / skip from the price one app pays for the whole multi against its break-even.
    With no app price there's nothing to judge yet: the card shows the price it needs."""
    price, p = m.get("best_book_price"), m.get("est_hit")
    if not price or not p:
        return {"verdict": "check", "value": None}
    value = round(p * price - 1, 4)
    return {"verdict": "value" if value >= VALUE_MARGIN else "skip" if value < SKIP_MARGIN else "marginal", "value": value}


def headline_for(multis: list[dict]) -> str:
    def where(m):
        return f"{m['multi']} on {odds.BOOK_NAMES.get(m['best_book'], m['best_book'])} ${m['best_book_price']:.2f} (needs ${m['break_even']:.2f})"
    seen, distinct = set(), []          # several named multis are often the same legs: say each once
    for m in multis:
        key = frozenset(m.get("leg_ids") or [m["multi"]])
        if key not in seen:
            seen.add(key)
            distinct.append(m)
    priced = [m for m in distinct if m.get("verdict") in ("value", "marginal", "skip")]
    good = sorted((m for m in priced if m["verdict"] == "value"), key=lambda m: -m["est_hit"])
    if good:
        return "Value today: " + "; ".join(where(m) for m in good[:3]) + "."
    if priced:
        closest = max(priced, key=lambda m: m["value"])
        return ("Skip today: no multi beats its break-even price by 5%. Closest: " + where(closest) + ".")
    top = next((m for m in multis if m["multi"] == "3-leg $2+"), None) or (multis[0] if multis else None)
    if not top:
        return ""
    return (f"{top['multi']}: {top['legs_text']} — about {top['est_hit']:.0%} to land, est. price ${top['est_price']:.2f}. "
            f"No app prices every leg yet: worth it at ${top['break_even']:.2f}+.")


def _multi_books(legs) -> dict:
    """A multi is placed with one bookie, so price it at each bookie that has every leg."""
    books = [l.extra.get("books") or {} for l in legs]
    none = {"book_prices": None, "best_book": None, "best_book_price": None}
    if not legs or not all(books):
        return none
    common = set(books[0]).intersection(*books[1:])
    by_book = {bk: round(_prod([b[bk] for b in books]), 2) for bk in common}
    if not by_book:
        return none
    bk = max(by_book, key=by_book.get)
    return {"book_prices": dict(sorted(by_book.items(), key=lambda x: -x[1])), "best_book": bk, "best_book_price": by_book[bk]}


def _prod(xs):
    out = 1.0
    for x in xs:
        out *= x
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["daily", "pretip", "late"], default="daily")
    ap.add_argument("--date", help="slate date YYYY-MM-DD (Melbourne); default today")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    now = datetime.now(espn.MELB)
    # Evening runs (after 4pm Melbourne) look ahead to the next morning's games.
    default_slate = now.date() + timedelta(days=1) if now.hour >= 16 else now.date()
    slate = date.fromisoformat(args.date) if args.date else default_slate
    if args.mode == "late":
        from . import late
        grade_live(now)  # today's slate, even after 4pm when `slate` has moved on to tomorrow
        late.run(slate, now)
        return
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
    prices = odds.empty()
    if args.mode == "pretip":  # keep the morning's prices
        from .late import restore_prices
        prices = load_prices(slate) or restore_prices(load_json("legs.json", []))

    # optional AI read of injury notes for players on today's teams
    teams_today = {g["home"] for g in upcoming} | {g["away"] for g in upcoming}
    ai_rows = [{"player": n.title(), "status": v["status"], "comment": v["comment"]}
               for n, v in inj.items() if v.get("comment") and v.get("status", "").lower() not in ("out",)]
    ai_inj = ai.read_injuries(ai_rows[:120]) if teams_today else {}

    found = analyse(slate, upcoming, ov, inj, ai_inj, prices)
    cands, chosen = select(found, upcoming)
    if args.mode == "daily" and (odds.enabled() or oddspapi.enabled()) and found:
        # second pass: price the games and stats the multis use first, then the other listed legs
        # (ESPN data is cached, so re-analysing is quick)
        names = {f"{g['away']} @ {g['home']}": (g["home_name"], g["away_name"]) for g in upcoming}
        in_multis = {i for k, v in chosen.items() if not k.startswith("_") for i in v}
        wanted: dict = {}
        for i, (l, _) in enumerate(cands):
            if l.stat != "win" and l.game in names:
                st = wanted.setdefault(names[l.game], {})
                st[l.stat] = st.get(l.stat, 0) + (10 if i in in_multis else 1)
        prices = odds.finalise(odds.merge(oddspapi.fetch([g["start"] for g in upcoming]), odds.fetch(upcoming, wanted)))
        if prices["books"] or prices["market"]:
            save_prices(slate, prices)
            found = analyse(slate, upcoming, ov, inj, ai_inj, prices)
            cands, chosen = select(found, upcoming)
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
            "start": g["start"].isoformat(),
            "favourite": g.get("favourite"), "spread": g.get("spread"), "total": g.get("total"),
            "fav_win_prob": fav_win, "status": "Analysed" if g in upcoming else "Started",
        })

    headline = headline_for(multis_out)
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
        "sources": {"stats": "ESPN", "odds": price_source(prices),
                    "injury_reading": "Claude" if ai.enabled() else "rules"},
        "updated_at": now.isoformat(),
        "inj_snapshot": {n: v["status"] for n, v in inj.items() if v.get("team") in slate_teams and v.get("status")},
        # how legs in one game move together, for the site's promo builder
        "links": {k: getattr(model, k) for k in ("SAME_GAME_REBOUNDS", "SAME_TEAM_POINTS", "SAME_GAME_OTHER", "WIN_VS_OPPONENT")},
    }
    if args.mode == "pretip":  # keep what the late checks have done so far today
        prev = load_json("report.json", {})
        for k in ("late_checked", "late_refreshes", "late_log", "lineup_checked"):
            if prev.get(k):
                report[k] = prev[k]
    if args.mode == "daily":
        save_json("h2h.json", odds.fetch_h2h(upcoming))  # for the promo page; 2 credits
    save_json("legs.json", legs_out)
    save_json("pool.json", pool_rows(found, match_winners(upcoming)))
    save_json("multis.json", multis_out)
    save_json("slate.json", slate_out)
    save_json("report.json", report)
    log.info("Wrote %d legs, %d multis. %s", len(legs_out), len(multis_out), changes or graded)


if __name__ == "__main__":
    main()

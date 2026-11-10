"""Backtest and tune the leg model on past games, and fit calibration from graded live legs.

    python -m punt_lab.nba.backtest

1. Fetch every current player's game logs for the last few seasons from ESPN
   (finished seasons are cached in punt_lab/nba/cache so later runs are quick).
2. Replay each game: using only what was known before tip-off (earlier games this
   season, last season as the prior, home/away split, the opponent's defence so far),
   ask the model how likely each line was, then check what actually happened.
3. Tune the settings in model.TUNABLE on the older season, and keep them only if they
   also predict better on the newer seasons (so it can't just memorise one season).
4. Fit a calibration curve per stat from the live graded legs (results.json), once
   there are enough of them.

The backtest can't replay the betting-market blend, teammate in/out shifts, pace,
injury minutes or head-to-head, because there's no free history of those. It tunes
the core of the model; the live calibration corrects whatever is left.

There's no free history of betting lines either, so each past game's line is stood in
for by the two teams' results up to that day (average winning margin, plus home court).
That is what the line adjustment and the blowout minutes cuts are fitted on.
"""
from __future__ import annotations

import bisect
import gzip
import json
import logging
import math
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from statistics import mean

from . import espn, model
from .build import DATA, LINES, MIN_GAMES, MIN_STAT_AVG, ROTATION_MINUTES, STATS, VENUE_GAMES, bucket, season_end_year
from .model import Leg, evaluate, norm_cdf

log = logging.getLogger("punt_lab.nba.backtest")
HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
OUT = DATA / "backtest.json"

KEEP_LOW, KEEP_HIGH = 0.35, 0.97     # lines scored: those the default model rates in this range (legs, plus the higher
                                     # "next line up" legs the long shot and mid-range use)
MIN_IMPROVEMENT = 0.002              # tuned settings must cut the newer seasons' log loss by at least 0.2%
LIVE_MIN_LEGS = 150                  # graded live legs needed before a stat gets a calibration curve
MIN_SAMPLES = 2000                   # per side (tuning / checking), or the run stops without changing anything
LIVE_SHRINK = 150                    # curve is pulled halfway to "no change" at this many legs
BINS = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0001]

MARGIN_MIN_SHARE = 0.5               # share of samples that need a known line stand-in before its settings are tuned
SIDE_GRID = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
CUT_GRID = [0.90, 0.93, 0.95, 0.97, 0.98, 1.0]
LINE_SETTINGS = ("SIDE_WEIGHT", "BLOWOUT_CUT", "UNDERDOG_CUT", "UNDERDOG_CUT_BIG")
CACHE_VERSION = 2                    # v2 adds each game's final margin

GRID = {
    "PRIOR_GAMES": [2, 4, 8, 12, 16, 24],
    "SPREAD_GAMES": [2, 4, 8, 16, 32, 64, 128],
    "RECENT_GAMES": [3, 5, 8, 10, 15, 20],
    "RECENT_WEIGHT": [1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 6.0],
    "OPP_WEIGHT": [0.0, 0.25, 0.5, 0.75, 1.0],
    "B2B_CUT_STAR": [0.86, 0.90, 0.93, 0.95, 0.97, 1.0],
    "B2B_CUT": [0.90, 0.94, 0.97, 1.0],
}
LINK_MIN_PAIRS = 1500                # pairs of legs needed before a same-game link is measured rather than assumed
LINKS = ("SAME_GAME_REBOUNDS", "SAME_TEAM_POINTS", "SAME_GAME_OTHER")
VERSIONS_FILE = HERE / "versions.json"  # every set of settings that has been in use, oldest first
SPREAD_STEPS = [0.8, 0.9, 1.0, 1.1, 1.2, 1.35]


# ---------- data ----------

def _players() -> dict[str, dict]:
    out = {}
    for tid in range(1, 31):
        for p in espn.team_roster(str(tid)):
            out[p["id"]] = p
    return out


def season_logs(year: int, players: dict[str, dict], current: int) -> dict[str, dict]:
    """{athlete_id: {name, pos, games: [...]}} for one season. Finished seasons are cached."""
    path = CACHE / f"gamelogs_v{CACHE_VERSION}_{year}.json.gz"
    cached = {}
    if year < current and path.exists():
        cached = json.loads(gzip.decompress(path.read_bytes()))
        missing = [pid for pid in players if pid not in cached]
    else:
        missing = list(players)
    for k, pid in enumerate(missing):
        lg = espn.last_season_log(pid, year)
        p = players[pid]
        cached[pid] = {"name": p["name"], "pos": p["pos"], "games": lg["games"]}
        if k % 100 == 99:
            log.info("season %d: %d / %d players fetched", year, k + 1, len(missing))
    if year < current and missing:
        CACHE.mkdir(exist_ok=True)
        path.write_bytes(gzip.compress(json.dumps(cached, separators=(",", ":")).encode()))
    return cached


def _played(games: list[dict]) -> list[dict]:
    gs = [g for g in games if g.get("min") and g.get("date")]
    gs.sort(key=lambda g: g["date"])
    return gs


class OppDefence:
    """Running per-game totals each team allowed, by position bucket and stat, before a date."""

    def __init__(self, logs: dict[str, dict]):
        per_game = defaultdict(float)   # (opp, date, pos, stat) -> total
        for p in logs.values():
            pos = bucket(p["pos"])
            for g in _played(p["games"]):
                if not g.get("opp"):
                    continue
                for stat, key in STATS.items():
                    per_game[(g["opp"], g["date"][:10], pos, stat)] += g.get(key) or 0
        series = defaultdict(list)       # (opp, pos, stat) -> [(date, total)]
        league = defaultdict(list)       # (pos, stat) -> [(date, total)]
        for (opp, d, pos, stat), v in per_game.items():
            series[(opp, pos, stat)].append((d, v))
            league[(pos, stat)].append((d, v))
        self.series = {k: self._cum(v) for k, v in series.items()}
        self.league = {k: self._cum(v) for k, v in league.items()}

    @staticmethod
    def _cum(rows):
        rows.sort()
        dates, sums, total = [], [], 0.0
        for d, v in rows:
            total += v
            dates.append(d)
            sums.append(total)
        return dates, sums

    @staticmethod
    def _avg_before(cum, d, min_n):
        dates, sums = cum
        n = bisect.bisect_left(dates, d)
        return sums[n - 1] / n if n >= min_n else None

    def factor(self, opp: str | None, pos: str, stat: str, d: str) -> float | None:
        if not opp or (opp, pos, stat) not in self.series:
            return None
        team = self._avg_before(self.series[(opp, pos, stat)], d, 3)
        lg = self._avg_before(self.league[(pos, stat)], d, 30)
        if not team or not lg:
            return None
        return round(max(0.85, min(1.15, team / lg)), 2)


class TeamRatings:
    """Each team's results before a date: the stand-in for the betting line (no free history of lines)."""

    def __init__(self, logs: dict[str, dict]):
        games = {}                       # (team, date) -> margin; every player in the game reports the same one
        for p in logs.values():
            for g in p["games"]:
                if g.get("team") and g.get("date") and g.get("margin") is not None:
                    games[(g["team"], g["date"][:10])] = g["margin"]
        by_team = defaultdict(list)
        for (team, d), m in games.items():
            by_team[team].append((d, m))
        self.series = {}
        for team, rows in by_team.items():
            rows.sort()
            self.series[team] = ([d for d, _ in rows], [m for _, m in rows])

    def before(self, team: str | None, d: str) -> float | None:
        if team not in self.series:
            return None
        dates, margins = self.series[team]
        n = bisect.bisect_left(dates, d)
        return model.rating(margins[:n]) if n >= MIN_GAMES else None

    def line(self, team: str | None, opp: str | None, d: str, home: bool | None) -> tuple[float | None, float | None]:
        """(expected margin tonight, how much tougher or easier that is than the team's usual), both in points."""
        mine, theirs = self.before(team, d), self.before(opp, d)
        if mine is None or theirs is None:
            return None, None
        edge = 0.0 if home is None else (model.HOME_EDGE if home else -model.HOME_EDGE)
        return round(mine - theirs + edge, 1), round(-theirs, 1)


def samples(year: int, logs: dict[str, dict], prior_logs: dict[str, dict]) -> list[dict]:
    """One sample per player-game-stat, using only games before it."""
    defence = OppDefence(logs)
    ratings = TeamRatings(logs)
    out = []
    for pid, p in logs.items():
        games = _played(p["games"])
        prior_games = _played((prior_logs.get(pid) or {}).get("games", []))
        pos = bucket(p["pos"])
        for i in range(MIN_GAMES, len(games)):
            before, g = games[:i], games[i]
            avg_min = mean(x["min"] for x in before)
            if avg_min < ROTATION_MINUTES:
                continue
            d = g["date"][:10]
            b2b = (date.fromisoformat(d) - date.fromisoformat(before[-1]["date"][:10])).days == 1
            expected, shift = ratings.line(g.get("team"), g.get("opp"), d, g.get("home"))
            for stat, key in STATS.items():
                vals = [x[key] for x in before if x.get(key) is not None]
                if len(vals) < MIN_GAMES or mean(vals) < MIN_STAT_AVG[stat] or g.get(key) is None:
                    continue
                prior_vals = [x[key] for x in prior_games if x.get(key) is not None]
                venue = [x[key] for x in before if x.get("home") is not None and x.get("home") == g.get("home") and x.get(key) is not None]
                venue_shift = 0.0
                if g.get("home") is not None and 2 <= len(venue) < len(vals):
                    venue_shift = (mean(venue) - mean(vals)) * len(venue) / (len(venue) + VENUE_GAMES)
                out.append({
                    "season": year, "stat": stat, "vals": vals, "actual": g[key],
                    "prior_mean": mean(prior_vals) if prior_vals else mean(vals), "prior_games": len(prior_vals),
                    "venue_shift": round(venue_shift, 2), "opp_factor": defence.factor(g.get("opp"), pos, stat, d),
                    "pid": pid, "date": d, "team": g.get("team"), "opp": g.get("opp"), "b2b": b2b, "avg_min": avg_min,
                    "expected": expected, "shift": shift,
                })
    return out


# ---------- scoring ----------

def _mean_sd(s: dict) -> tuple[float, float]:
    minutes = s.get("avg_min")
    if minutes:
        if s.get("b2b"):
            minutes *= model.B2B_CUT_STAR if s["avg_min"] >= 30 else model.B2B_CUT
        minutes *= model.blowout_cut(s.get("expected"))
    leg = Leg(leg_id="bt", player="bt", team="", game="", stat=s["stat"], line=LINES[s["stat"]][0], log=s["vals"],
              prior_mean=s["prior_mean"], prior_games=s["prior_games"], venue_shift=s["venue_shift"],
              opp_factor=s["opp_factor"], avg_minutes=s.get("avg_min"), proj_minutes=minutes,
              side_factor=model.side_factor(s["stat"], s["shift"]) if s.get("shift") is not None else 1.0)
    r = evaluate(leg)
    return r.proj_mean, r.proj_sd


def _prob(line: float, m: float, sd: float) -> float:
    return 1 - norm_cdf((line - 0.5 - m) / sd)


def choose_lines(data: list[dict]) -> None:
    """Fix which lines each sample is scored on, using the default settings, so tuning can't move the goalposts."""
    for s in data:
        m, sd = _mean_sd(s)
        s["lines"] = [ln for ln in LINES[s["stat"]] if KEEP_LOW <= _prob(ln, m, sd) <= KEEP_HIGH]


def score(data: list[dict], curves: dict | None = None) -> dict:
    """Log loss and Brier score over every (sample, line), plus calibration bins."""
    ll = br = 0.0
    n = 0
    bins = [[0, 0.0, 0] for _ in BINS[:-1]]  # count, sum predicted, landed
    for s in data:
        if not s["lines"]:
            continue
        m, sd = _mean_sd(s)
        for ln in s["lines"]:
            p = _prob(ln, m, sd)
            if curves:
                p = model._curve(model.curve_for(curves, s["stat"], ln), p)
            p = min(max(p, 1e-4), 1 - 1e-4)
            y = 1.0 if s["actual"] >= ln else 0.0
            ll -= y * math.log(p) + (1 - y) * math.log(1 - p)
            br += (p - y) ** 2
            n += 1
            b = bisect.bisect_right(BINS, p) - 1
            if 0 <= b < len(bins):
                bins[b][0] += 1
                bins[b][1] += p
                bins[b][2] += y
    return {"n": n, "log_loss": ll / n if n else None, "brier": br / n if n else None,
            "bins": [{"from": BINS[i], "to": min(1.0, BINS[i + 1]), "n": c, "predicted": round(sp / c, 4), "actual": round(y / c, 4)}
                     for i, (c, sp, y) in enumerate(bins) if c]}


def current_settings() -> dict:
    return {k: (dict(getattr(model, k)) if isinstance(getattr(model, k), dict) else getattr(model, k)) for k in model.TUNABLE}


def apply_settings(settings: dict) -> None:
    for k, v in settings.items():
        setattr(model, k, dict(v) if isinstance(v, dict) else v)


def has_lines(data: list[dict]) -> bool:
    """Enough past games have a line stand-in (needs final scores from ESPN) to fit the line settings."""
    return bool(data) and sum(1 for s in data if s.get("shift") is not None) >= MARGIN_MIN_SHARE * len(data)


def tune(train: list[dict], rounds: int = 3) -> tuple[dict, float]:
    """Coordinate search: try each setting's candidate values in turn, keep whatever lowers log loss."""
    best = current_settings()
    apply_settings(best)
    best_ll = score(train)["log_loss"]
    for _ in range(rounds):
        improved = False
        steps: list[tuple[str, str | None, list]] = [(k, None, v) for k, v in GRID.items()]
        steps += [("SPREAD", stat, [round(best["SPREAD"][stat] * f, 3) for f in SPREAD_STEPS]) for stat in best["SPREAD"]]
        for name, stat, values in steps:
            for v in values:
                trial = json.loads(json.dumps(best))
                if stat:
                    trial[name][stat] = v
                else:
                    trial[name] = v
                if trial == best:
                    continue
                apply_settings(trial)
                ll = score(train)["log_loss"]
                if ll < best_ll - 1e-6:
                    best, best_ll, improved = trial, ll, True
            apply_settings(best)
            log.info("tuned %s%s -> log loss %.4f", name, f"[{stat}]" if stat else "", best_ll)
        if not improved:
            break
    return best, best_ll


def tune_lines(train: list[dict], test: list[dict], base: dict) -> tuple[dict, dict]:
    """Fit the line settings (how much tonight's line moves a prediction, and the blowout minutes cuts)
    on the older season, then keep whichever of three does best on the newer seasons: the fitted
    values, the ones in use, or no adjustment at all. Returns (settings, a report for the site)."""
    in_use = {k: json.loads(json.dumps(base[k])) for k in LINE_SETTINGS}
    neutral = {"SIDE_WEIGHT": {stat: 0.0 for stat in base["SIDE_WEIGHT"]},
               "BLOWOUT_CUT": 1.0, "UNDERDOG_CUT": 1.0, "UNDERDOG_CUT_BIG": 1.0}

    def loss(data, lines):
        apply_settings({**base, **lines})
        return score(data)["log_loss"]

    fitted = json.loads(json.dumps(in_use))
    best_ll = loss(train, fitted)
    for _ in range(2):
        improved = False
        steps = [("SIDE_WEIGHT", stat, SIDE_GRID) for stat in fitted["SIDE_WEIGHT"]]
        steps += [(k, None, CUT_GRID) for k in LINE_SETTINGS if k != "SIDE_WEIGHT"]
        for name, stat, values in steps:
            for v in values:
                trial = json.loads(json.dumps(fitted))
                if stat:
                    trial[name][stat] = v
                else:
                    trial[name] = v
                if trial == fitted:
                    continue
                ll = loss(train, trial)
                if ll < best_ll - 1e-6:
                    fitted, best_ll, improved = trial, ll, True
        if not improved:
            break
    options = {"fitted": fitted, "in use": in_use, "none": neutral}
    scores = {k: loss(test, v) for k, v in options.items()}
    pick_ = min(scores, key=scores.get)
    log.info("line settings: newer seasons log loss fitted %.5f, in use %.5f, none %.5f -> %s",
             scores["fitted"], scores["in use"], scores["none"], pick_)
    apply_settings(base)
    return options[pick_], {"chosen": pick_, "log_loss": {k: round(v, 5) for k, v in scores.items()}, "fitted": fitted}


# ---------- correction curve ----------

def fit_curves(data: list[dict], min_n: int = 200, by_line: bool = False) -> dict[str, list[list[float]]]:
    """Per stat: predicted vs actually landed in 5% bins, made monotone. Applied to every leg before
    the live calibration, so the percentages match what happened across past seasons.
    With by_line, a line with enough history ("points:10") also gets a curve of its own: low lines
    miss more than the bell curve says (a 16-a-game scorer has more single-figure nights than it allows)."""
    rows = defaultdict(list)
    for s in data:
        if not s.get("lines"):
            continue
        m, sd = _mean_sd(s)
        for ln in s["lines"]:
            row = (_prob(ln, m, sd), 1.0 if s["actual"] >= ln else 0.0)
            rows[s["stat"]].append(row)
            if by_line:
                rows[f"{s['stat']}:{int(ln)}"].append(row)
    curves = {}
    for stat, rs in rows.items():
        pts = []
        for lo, hi in zip(BINS, BINS[1:]):
            g = [(p, y) for p, y in rs if lo <= p < hi]
            if len(g) >= min_n:
                pts.append([mean(p for p, _ in g), mean(y for _, y in g), len(g)])
        pts = _pav(pts)
        if len(pts) >= 3:
            curves[stat] = [[round(x, 4), round(y, 4)] for x, y, _ in pts]
    return curves


# ---------- same-game links ----------

def measure_links(data: list[dict]) -> dict[str, dict]:
    """How much two legs in the same game land or miss together, beyond what their chances say.
    For each pair of different players in one game (each at the line nearest 75%), the average of
    (landed - chance) / sd multiplied together estimates the correlation the multi maths uses."""
    games = defaultdict(list)
    for s in data:
        if not s.get("lines") or not s.get("team") or not s.get("opp"):
            continue
        m, sd = _mean_sd(s)
        ln, p = min(((ln, _prob(ln, m, sd)) for ln in s["lines"]), key=lambda x: abs(x[1] - 0.75))
        p = min(max(p, 0.02), 0.98)
        z = ((1.0 if s["actual"] >= ln else 0.0) - p) / math.sqrt(p * (1 - p))
        games[(s["date"], tuple(sorted((s["team"], s["opp"]))))].append((s["pid"], s["team"], s["stat"], z))
    sums = defaultdict(float)
    counts = defaultdict(int)
    for rows in games.values():
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                a, b = rows[i], rows[j]
                if a[0] == b[0]:
                    continue  # same player: multis never use two of his legs
                if a[2] == "rebounds" and b[2] == "rebounds":
                    k = "SAME_GAME_REBOUNDS"
                elif a[1] == b[1] and a[2] == "points" and b[2] == "points":
                    k = "SAME_TEAM_POINTS"
                else:
                    k = "SAME_GAME_OTHER"
                sums[k] += a[3] * b[3]
                counts[k] += 1
    return {k: {"pairs": counts[k], "r": round(max(-0.3, min(0.3, sums[k] / counts[k])), 3) if counts[k] else None}
            for k in LINKS}


# ---------- live calibration ----------

def _stat_of(row: dict) -> str | None:
    text = f"{row.get('market', '')} {row.get('stat', '')}".lower()
    for stat in STATS:
        if stat in text:
            return stat
    return None


def _pav(points: list[list[float]]) -> list[list[float]]:
    """Pool-adjacent-violators: make the actual rate never fall as the predicted chance rises. points = [pred, actual, weight]."""
    blocks = [list(p) for p in points]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][1] > blocks[i + 1][1]:
            a, b = blocks[i], blocks[i + 1]
            w = a[2] + b[2]
            blocks[i] = [(a[0] * a[2] + b[0] * b[2]) / w, (a[1] * a[2] + b[1] * b[2]) / w, w]
            del blocks[i + 1]
            i = max(i - 1, 0)
        else:
            i += 1
    return blocks


def live_calibration(results: list[dict]) -> tuple[dict, dict]:
    """Calibration curves per stat from graded live legs, using the raw (pre-calibration) chance."""
    by_stat = defaultdict(list)
    for r in results:
        stat, landed = _stat_of(r), r.get("landed")
        p = r.get("raw_prob") or r.get("model_prob")
        if stat and landed in (True, False) and p is not None:
            by_stat[stat].append((p, 1.0 if landed else 0.0))
    curves, summary = {}, {}
    for stat, rows in by_stat.items():
        summary[stat] = {"legs": len(rows), "landed": round(mean(y for _, y in rows), 4), "predicted": round(mean(p for p, _ in rows), 4)}
        if len(rows) < LIVE_MIN_LEGS:
            continue
        shrink = len(rows) / (len(rows) + LIVE_SHRINK)
        pts = []
        for lo, hi in zip(BINS, BINS[1:]):
            g = [(p, y) for p, y in rows if lo <= p < hi]
            if len(g) >= 10:
                pts.append([mean(p for p, _ in g), mean(y for _, y in g), len(g)])
        pts = _pav(pts)
        if len(pts) >= 2:
            curves[stat] = [[round(x, 4), round(x + shrink * (y - x), 4)] for x, y, _ in pts]
    return curves, summary


# ---------- version history ----------

def _curve_kind(curves: dict) -> str:
    return "none" if not curves else ("line" if any(":" in k for k in curves) else "stat")


def load_versions() -> list[dict]:
    try:
        return json.loads(VERSIONS_FILE.read_text())
    except (OSError, ValueError):
        return []


def record_version(versions: list[dict], day: str, settings: dict, curves: dict, defaults: dict) -> list[dict]:
    """Add this week's settings to the history if they'd change a single leg's chance. Same-game links
    only matter to multis, so they don't make a new version on this chart."""
    entry = {"date": day, "settings": {k: v for k, v in settings.items() if k not in LINKS and v != defaults.get(k)},
             "curve": _curve_kind(curves)}
    last = versions[-1] if versions else {"settings": {}, "curve": "none"}
    if (entry["settings"], entry["curve"]) != (last["settings"], last["curve"]):
        versions = versions + [entry]
    return versions


def score_versions(versions: list[dict], defaults: dict, train: list[dict], test: list[dict]) -> list[dict]:
    """Re-score the hand-set model and every past version on this week's legs, the same way: settings
    applied, correction curve (if that version used one) fitted on the older season, checked on the newer."""
    out = []
    for v in [{"date": None, "settings": {}, "curve": "none"}] + versions:
        apply_settings({**defaults, **v["settings"]})
        curves = fit_curves(train, by_line=v["curve"] == "line") if v["curve"] != "none" else {}
        r = score(test, curves)
        out.append({"date": v["date"], "curve": v["curve"], "n": r["n"], "log_loss": r["log_loss"], "brier": r["brier"], "bins": r["bins"]})
    return out


# ---------- main ----------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    today = datetime.now(espn.MELB).date()
    cur = season_end_year(today)
    model.CALIBRATE = False
    defaults = json.loads(json.dumps(model.DEFAULTS))   # the hand-set values in model.py
    previous = current_settings()                       # what's in use now (last week's tune, or the defaults)

    players = _players()
    log.info("%d players on current rosters", len(players))
    logs = {y: season_logs(y, players, cur) for y in (cur - 3, cur - 2, cur - 1, cur)}
    train = samples(cur - 2, logs[cur - 2], logs[cur - 3])
    test = samples(cur - 1, logs[cur - 1], logs[cur - 2]) + samples(cur, logs[cur], logs[cur - 1])
    log.info("samples: train %d, test %d; with a final score: train %d, test %d", len(train), len(test),
             sum(1 for s in train if s.get("shift") is not None), sum(1 for s in test if s.get("shift") is not None))
    if len(train) < MIN_SAMPLES or len(test) < MIN_SAMPLES:
        raise SystemExit("not enough backtest data from ESPN; leaving the model unchanged")

    apply_settings(defaults)
    choose_lines(train)  # always scored on the lines the hand-set model would consider, so weeks compare fairly
    choose_lines(test)
    before = {"train": score(train), "test": score(test)}
    apply_settings(previous)
    prev_test = score(test)["log_loss"]

    tuned, _ = tune(train)  # starts from what's in use now
    apply_settings(tuned)
    new_test = score(test)["log_loss"]
    better = prev_test - new_test >= MIN_IMPROVEMENT * prev_test
    final = tuned if better else previous
    apply_settings(final)
    line_report = {"chosen": "not fitted", "reason": "ESPN gave no final scores for past games"}
    if has_lines(train) and has_lines(test):
        line_settings, line_report = tune_lines(train, test, final)
        final = {**final, **line_settings}
        apply_settings(final)
    else:
        log.info("line settings not fitted: only %d of %d past games have a final score",
                 sum(1 for s in train if s.get("shift") is not None), len(train))
    links = measure_links(train + test)
    for k, v in links.items():
        if v["r"] is not None and v["pairs"] >= LINK_MIN_PAIRS:
            final[k] = v["r"]
            log.info("measured %s = %+.3f from %d pairs (was %+.3f)", k, v["r"], v["pairs"], defaults[k])
    apply_settings(final)
    # correction curves: fitted on the older season, checked on the newer ones. Per-line curves are
    # used only if they beat the per-stat ones there.
    uncorrected = score(test)
    options = {False: fit_curves(train), True: fit_curves(train, by_line=True)}
    scores = {k: score(test, c)["log_loss"] for k, c in options.items()}
    by_line = scores[True] < scores[False]
    curves = options[by_line]
    after = {"train": score(train, curves), "test": score(test, curves)}
    if after["test"]["log_loss"] >= uncorrected["log_loss"]:
        log.info("correction curve didn't help on the newer seasons; not used")
        curves, after = {}, {"train": score(train), "test": uncorrected}
    else:
        log.info("correction curve (%s): newer seasons log loss %.4f -> %.4f (per stat %.4f, per line %.4f)",
                 "per line" if by_line else "per stat", uncorrected["log_loss"], after["test"]["log_loss"], scores[False], scores[True])
        curves = fit_curves(train + test, by_line=by_line)  # then refit on everything for use
    applied = final != defaults
    log.info("newer seasons log loss: hand-set %.4f, in use %.4f, new tune %.4f (%s)", before["test"]["log_loss"],
             prev_test, new_test, "switched to new tune" if better else "kept the settings in use")

    versions = load_versions()
    if applied:
        versions = record_version(versions, today.isoformat(), final, curves, defaults)
    history = score_versions(versions, defaults, train, test)
    apply_settings(final)
    VERSIONS_FILE.write_text(json.dumps(versions, indent=1))

    try:
        live = json.loads((DATA / "results.json").read_text())
    except (OSError, ValueError):
        live = []
    live_curves, live_summary = live_calibration(live)

    (model.TUNED_FILE).write_text(json.dumps({
        "updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "settings": final if applied else {}, "calibration": live_curves, "backtest_curve": curves}, indent=1))
    OUT.write_text(json.dumps({
        "updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seasons": {"tuned_on": f"{cur - 3}-{str(cur - 2)[2:]}", "checked_on": f"{cur - 2}-{str(cur - 1)[2:]} and {cur - 1}-{str(cur)[2:]} so far"},
        "samples": {"train": len(train), "test": len(test)},
        "applied": applied, "defaults": defaults, "tuned": final, "changed_this_week": better, "links": links,
        "lines": {**line_report, "fitted_on_scores": has_lines(train), "train_share": round(sum(1 for s in train if s.get("shift") is not None) / len(train), 3),
                  "curves_by_line": sorted(k for k in curves if ":" in k)},
        "before": before, "after": after, "versions": history,
        "live": {"summary": live_summary, "curves": live_curves, "min_legs": LIVE_MIN_LEGS},
        "backtest_curve": curves,
    }, indent=1))


if __name__ == "__main__":
    main()

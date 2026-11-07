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

KEEP_LOW, KEEP_HIGH = 0.50, 0.97     # lines scored: those the default model rates in this range (the ones that become legs)
MIN_IMPROVEMENT = 0.002              # tuned settings must cut the newer seasons' log loss by at least 0.2%
LIVE_MIN_LEGS = 150                  # graded live legs needed before a stat gets a calibration curve
MIN_SAMPLES = 2000                   # per side (tuning / checking), or the run stops without changing anything
LIVE_SHRINK = 150                    # curve is pulled halfway to "no change" at this many legs
BINS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0001]

GRID = {
    "PRIOR_GAMES": [2, 4, 8, 12, 16, 24],
    "SPREAD_GAMES": [2, 4, 8, 16, 32],
    "RECENT_GAMES": [3, 5, 8, 10],
    "RECENT_WEIGHT": [1.0, 1.25, 1.5, 2.0, 3.0],
    "OPP_WEIGHT": [0.0, 0.25, 0.5, 0.75, 1.0],
}
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
    path = CACHE / f"gamelogs_{year}.json.gz"
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


def samples(year: int, logs: dict[str, dict], prior_logs: dict[str, dict]) -> list[dict]:
    """One sample per player-game-stat, using only games before it."""
    defence = OppDefence(logs)
    out = []
    for pid, p in logs.items():
        games = _played(p["games"])
        prior_games = _played((prior_logs.get(pid) or {}).get("games", []))
        pos = bucket(p["pos"])
        for i in range(MIN_GAMES, len(games)):
            before, g = games[:i], games[i]
            if mean(x["min"] for x in before) < ROTATION_MINUTES:
                continue
            d = g["date"][:10]
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
                })
    return out


# ---------- scoring ----------

def _mean_sd(s: dict) -> tuple[float, float]:
    leg = Leg(leg_id="bt", player="bt", team="", game="", stat=s["stat"], line=LINES[s["stat"]][0], log=s["vals"],
              prior_mean=s["prior_mean"], prior_games=s["prior_games"], venue_shift=s["venue_shift"],
              opp_factor=s["opp_factor"])
    r = evaluate(leg)
    return r.proj_mean, r.proj_sd


def _prob(line: float, m: float, sd: float) -> float:
    return 1 - norm_cdf((line - 0.5 - m) / sd)


def choose_lines(data: list[dict]) -> None:
    """Fix which lines each sample is scored on, using the default settings, so tuning can't move the goalposts."""
    for s in data:
        m, sd = _mean_sd(s)
        s["lines"] = [ln for ln in LINES[s["stat"]] if KEEP_LOW <= _prob(ln, m, sd) <= KEEP_HIGH]


def score(data: list[dict]) -> dict:
    """Log loss and Brier score over every (sample, line), plus calibration bins."""
    ll = br = 0.0
    n = 0
    bins = [[0, 0.0, 0] for _ in BINS[:-1]]  # count, sum predicted, landed
    for s in data:
        if not s["lines"]:
            continue
        m, sd = _mean_sd(s)
        for ln in s["lines"]:
            p = min(max(_prob(ln, m, sd), 1e-4), 1 - 1e-4)
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


def tune(train: list[dict], rounds: int = 2) -> tuple[dict, float]:
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
                    trial["SPREAD"][stat] = v
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


# ---------- main ----------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    today = datetime.now(espn.MELB).date()
    cur = season_end_year(today)
    model.CALIBRATE = False
    defaults = current_settings()

    players = _players()
    log.info("%d players on current rosters", len(players))
    logs = {y: season_logs(y, players, cur) for y in (cur - 3, cur - 2, cur - 1, cur)}
    train = samples(cur - 2, logs[cur - 2], logs[cur - 3])
    test = samples(cur - 1, logs[cur - 1], logs[cur - 2]) + samples(cur, logs[cur], logs[cur - 1])
    log.info("samples: train %d, test %d", len(train), len(test))
    if len(train) < MIN_SAMPLES or len(test) < MIN_SAMPLES:
        raise SystemExit("not enough backtest data from ESPN; leaving the model unchanged")

    apply_settings(defaults)
    choose_lines(train)
    choose_lines(test)
    before = {"train": score(train), "test": score(test)}

    tuned, _ = tune(train)
    apply_settings(tuned)
    after = {"train": score(train), "test": score(test)}
    gain = before["test"]["log_loss"] - after["test"]["log_loss"]
    applied = gain >= MIN_IMPROVEMENT * before["test"]["log_loss"]
    final = tuned if applied else defaults
    log.info("newer seasons log loss %.4f -> %.4f (%s)", before["test"]["log_loss"], after["test"]["log_loss"],
             "applied" if applied else "not enough better; keeping defaults")

    try:
        live = json.loads((DATA / "results.json").read_text())
    except (OSError, ValueError):
        live = []
    curves, live_summary = live_calibration(live)

    (model.TUNED_FILE).write_text(json.dumps({
        "updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "settings": final if applied else {}, "calibration": curves}, indent=1))
    OUT.write_text(json.dumps({
        "updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seasons": {"tuned_on": f"{cur - 3}-{str(cur - 2)[2:]}", "checked_on": f"{cur - 2}-{str(cur - 1)[2:]} and {cur - 1}-{str(cur)[2:]} so far"},
        "samples": {"train": len(train), "test": len(test)},
        "applied": applied, "defaults": defaults, "tuned": tuned,
        "before": before, "after": after,
        "live": {"summary": live_summary, "curves": curves, "min_legs": LIVE_MIN_LEGS},
    }, indent=1))


if __name__ == "__main__":
    main()

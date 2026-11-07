"""The leg and multi model. Mirrors the NBA Daily Multi dashboard exactly."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, pstdev

SPREAD = {"points": 0.28, "rebounds": 0.32, "assists": 0.38, "threes": 0.55}
PRIOR_GAMES = 8          # last season counts as this many games...
PRIOR_FULL_SEASON = 40   # ...if the player played at least this many games then
SPREAD_GAMES = 8         # weight of the typical spread when estimating SD
CONTEXT_GAMES = 5        # situation shift is halfway with this many matching games
MARKET_WEIGHT = 0.6      # share of the final mean taken from the betting market
OPP_WEIGHT = 0.5         # this season's defence vs position
H2H_WEIGHT = 0.5         # last season's head-to-head (also scaled by roster overlap)
H2H_GAMES = 4            # head-to-head is halfway with this many games
RECENT_GAMES = 5         # the most recent games...
RECENT_WEIGHT = 1.5      # ...count this much more than older ones

# Settings the weekly backtest is allowed to tune (see backtest.py). Values above are the
# hand-set defaults; tuned.json, when present, replaces them.
TUNABLE = ("SPREAD", "PRIOR_GAMES", "SPREAD_GAMES", "RECENT_GAMES", "RECENT_WEIGHT", "OPP_WEIGHT")
TUNED_FILE = Path(__file__).with_name("tuned.json")
CALIBRATION: dict[str, list[list[float]]] = {}  # stat -> [[model chance, actual landed], ...] from graded live legs
CALIBRATE = True  # the backtest turns this off to see the raw model


def load_tuned(path: Path = TUNED_FILE) -> dict:
    global CALIBRATION
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    g = globals()
    for k, v in (data.get("settings") or {}).items():
        if k in TUNABLE:
            g[k] = dict(v) if isinstance(v, dict) else v
    CALIBRATION = data.get("calibration") or {}
    return data


def calibrate(stat: str, p: float) -> float:
    """Map a raw model chance through the curve fitted on graded live legs (straight-line between points)."""
    pts = CALIBRATION.get(stat) if CALIBRATE else None
    if not pts:
        return p
    if p <= pts[0][0]:
        return p + (pts[0][1] - pts[0][0])
    if p >= pts[-1][0]:
        return min(0.995, p + (pts[-1][1] - pts[-1][0]))
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= p <= x1:
            return y0 + (y1 - y0) * (p - x0) / (x1 - x0) if x1 > x0 else y0
    return p


# Same-game links between legs in a multi
SAME_GAME_REBOUNDS = -0.15
SAME_TEAM_POINTS = -0.05
SAME_GAME_OTHER = 0.05


def norm_cdf(z: float) -> float:
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def norm_inv(p: float) -> float:
    lo, hi = -8.0, 8.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if norm_cdf(mid) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


@dataclass
class Leg:
    leg_id: str
    player: str
    team: str
    game: str
    stat: str
    line: float
    log: list[float]
    prior_mean: float
    prior_games: int
    market: str = ""
    odds: float | None = None
    context_label: str | None = None
    context_log: list[float] | None = None
    h2h_log: list[float] | None = None
    roster_overlap: float | None = None
    venue_shift: float = 0.0       # home/away adjustment, in the stat's units
    pace_factor: float = 1.0       # tonight's expected game total vs average
    opp_factor: float | None = None
    avg_minutes: float | None = None
    proj_minutes: float | None = None
    market_line: float | None = None
    market_p_over: float | None = None
    market_source: str | None = None
    note: str = ""
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.market:
            label = {"points": "points", "rebounds": "rebounds", "assists": "assists", "threes": "threes"}[self.stat]
            self.market = f"{int(self.line)}+ {label}"


@dataclass
class LegResult:
    hit_rate: float
    context_shift: float
    matchup: float
    minutes_factor: float
    proj_mean: float
    proj_sd: float
    market_mean: float | None
    fair_price: float | None
    model_prob: float
    implied: float | None
    edge: float | None
    raw_prob: float | None = None  # model chance before the live calibration curve


def evaluate(leg: Leg) -> LegResult:
    xs = leg.log
    n = len(xs)
    m = mean(xs)
    weights = [RECENT_WEIGHT if i >= n - int(RECENT_GAMES) else 1.0 for i in range(n)]
    m_recent = sum(w * x for w, x in zip(weights, xs)) / sum(weights)
    sd = pstdev(xs) if n > 1 else 0.0
    k = PRIOR_GAMES * min(1.0, leg.prior_games / PRIOR_FULL_SEASON)
    blended = (n * m_recent + k * leg.prior_mean) / (n + k)

    context_shift = 0.0
    if leg.context_log:
        c = len(leg.context_log)
        context_shift = (mean(leg.context_log) - m) * c / (c + CONTEXT_GAMES)

    opp_part = OPP_WEIGHT * (leg.opp_factor - 1) if leg.opp_factor is not None else 0.0
    h2h_part = 0.0
    if leg.h2h_log and leg.prior_mean and leg.roster_overlap is not None:
        g = len(leg.h2h_log)
        h2h_part = H2H_WEIGHT * leg.roster_overlap * (mean(leg.h2h_log) / leg.prior_mean - 1) * g / (g + H2H_GAMES)
    matchup = 1 + opp_part + h2h_part

    minutes_factor = leg.proj_minutes / leg.avg_minutes if leg.avg_minutes and leg.proj_minutes else 1.0
    proj_mean = (blended + context_shift + leg.venue_shift) * matchup * minutes_factor * leg.pace_factor

    typical = SPREAD[leg.stat] * proj_mean
    proj_sd = math.sqrt((n * sd * sd + SPREAD_GAMES * typical * typical) / (n + SPREAD_GAMES))
    proj_sd = max(proj_sd, 0.5)

    market_mean = fair_price = None
    final_mean = proj_mean
    if leg.market_line is not None and leg.market_p_over is not None:
        market_mean = leg.market_line - proj_sd * norm_inv(1 - leg.market_p_over)
        market_prob = 1 - norm_cdf((leg.line - 0.5 - market_mean) / proj_sd)
        fair_price = 1 / market_prob if market_prob > 0 else None
        final_mean = MARKET_WEIGHT * market_mean + (1 - MARKET_WEIGHT) * proj_mean

    raw_prob = 1 - norm_cdf((leg.line - 0.5 - final_mean) / proj_sd)
    model_prob = calibrate(leg.stat, raw_prob)
    hit_rate = sum(1 for x in xs if x >= leg.line) / n
    implied = 1 / leg.odds if leg.odds else None
    edge = model_prob - implied if implied is not None else None
    return LegResult(hit_rate, context_shift, matchup, minutes_factor, final_mean, proj_sd,
                     market_mean, fair_price, model_prob, implied, edge, raw_prob)


def pair_link(a: Leg, b: Leg) -> float:
    if a.game != b.game:
        return 0.0
    if a.stat == "win" or b.stat == "win":
        return SAME_GAME_OTHER if a.stat != b.stat else 0.0
    if a.stat == "rebounds" and b.stat == "rebounds":
        return SAME_GAME_REBOUNDS
    if a.team == b.team and a.stat == "points" and b.stat == "points":
        return SAME_TEAM_POINTS
    return SAME_GAME_OTHER


def multi_chance(legs: list[Leg], probs: list[float]) -> float:
    p = 1.0
    for x in probs:
        p *= x
    adj = 1.0
    for i in range(len(legs)):
        for j in range(i + 1, len(legs)):
            r = pair_link(legs[i], legs[j])
            if r and probs[i] > 0 and probs[j] > 0:
                adj *= max(0.5, 1 + r * math.sqrt((1 - probs[i]) * (1 - probs[j]) / (probs[i] * probs[j])))
    return min(1.0, p * adj)


def american_to_prob(odds: float) -> float:
    return -odds / (-odds + 100) if odds < 0 else 100 / (odds + 100)


def no_vig_over(over_american: float, under_american: float) -> float:
    o, u = american_to_prob(over_american), american_to_prob(under_american)
    return o / (o + u)


load_tuned()

"""Choosing the three multis from the candidate legs."""
from __future__ import annotations

import math
from itertools import combinations

from .model import Leg, LegResult, multi_chance

MIN_PRICE = 1.20          # Sam rarely takes a leg shorter than this
MAX_UNPRICED_MODEL = 0.83 # with no price known, a leg this likely would be priced under $1.20
BOOKIE_MARGIN = 0.92      # assumed Sportsbet price = this x fair, when no price is known
POOL = 16                 # legs considered when searching combinations
MIN_MULTI = 2.00          # the "$2+" multis must be estimated to pay at least this
LONG_TARGET = 50.0        # long-shot multi: estimated Sportsbet price to reach
LONG_LEGS = range(8, 13)  # 8 to 12 legs
LONG_MIN_PRICE = 1.15     # shortest leg allowed in the long shot
WIDE_POOL = 20            # $2+ multis search the top legs plus up to this many step-up (next line up) legs


def price_of(leg: Leg, res: LegResult) -> tuple[float | None, str]:
    if leg.odds:
        return leg.odds, "odds"
    if res.fair_price:
        return res.fair_price, "fair"
    return None, "none"


def price_ok(leg: Leg, res: LegResult) -> bool:
    price, kind = price_of(leg, res)
    if price is not None:
        return price >= MIN_PRICE
    return res.model_prob <= MAX_UNPRICED_MODEL


def eligible(leg: Leg, res: LegResult) -> bool:
    if leg.extra.get("risky"):
        return False  # questionable / game-time decision: listed, but kept out of multis
    return price_ok(leg, res)


def _ok_combo(legs: list[Leg]) -> bool:
    players = [l.player for l in legs]
    if len(set(players)) != len(players):
        return False
    for a, b in combinations(legs, 2):
        if a.game == b.game and a.stat == "rebounds" and b.stat == "rebounds":
            return False  # two players chasing the same boards
    return True


def _assumed_price(leg: Leg, res: LegResult) -> float:
    price, kind = price_of(leg, res)
    if kind == "odds":
        return price
    if kind == "fair":
        return price * BOOKIE_MARGIN
    return BOOKIE_MARGIN / res.model_prob


def choose(cands: list[tuple[Leg, LegResult]]) -> dict[str, list[int]]:
    """Returns {multi name: [indexes into cands]}"""
    every = [i for i, (l, r) in enumerate(cands) if eligible(l, r)]
    every.sort(key=lambda i: -cands[i][1].model_prob)
    idx = [i for i in every if not cands[i][0].extra.get("step_up")]  # step-up legs only feed the $2+ multis
    pool = idx[:POOL]
    out: dict[str, list[int]] = {}

    def chance(combo):
        return multi_chance([cands[i][0] for i in combo], [cands[i][1].model_prob for i in combo])

    def ev(combo):
        p = chance(combo)
        price = 1.0
        for i in combo:
            price *= _assumed_price(*cands[i])
        return p * price, p

    # Safest: highest chance all four land
    best = None
    for combo in combinations(pool, 4):
        if _ok_combo([cands[i][0] for i in combo]):
            c = chance(combo)
            if best is None or c > best[0]:
                best = (c, combo)
    if best:
        out["Safest"] = list(best[1])

    # Best value: highest expected return, keeping a sensible chance of landing
    best = None
    for combo in combinations(pool, 4):
        if not _ok_combo([cands[i][0] for i in combo]):
            continue
        e, p = ev(combo)
        if p >= 0.30 and (best is None or e > best[0]):
            best = (e, combo)
    if best:
        out["Best value"] = list(best[1])

    # 3-leg value: every leg must be worth it on its own (model x price > 1)
    worth = [i for i in pool if price_of(*cands[i])[0] and cands[i][1].model_prob * _assumed_price(*cands[i]) > 1]

    def best_three(search, min_p):
        best = None
        for combo in combinations(search, 3):
            if not _ok_combo([cands[i][0] for i in combo]):
                continue
            e, p = ev(combo)
            if p >= min_p and (best is None or e > best[0]):
                best = (e, combo)
        return best

    all_worth = True
    best = best_three(worth, 0.45) or best_three(worth, 0.0)
    if best is None:  # not enough worth-it legs that fit together: fill from the pool
        all_worth = False
        best = best_three(pool, 0.45) or best_three(pool, 0.0)
    if best:
        out["3-leg value"] = list(best[1])
    # $2+ multis: highest chance all land, with the estimated Sportsbet price at least $2.00
    wide = idx[:WIDE_POOL] + [i for i in every if cands[i][0].extra.get("step_up")][:WIDE_POOL]

    def multi_price(combo):
        price = 1.0
        for i in combo:
            price *= _assumed_price(*cands[i])
        return price

    for n, name in ((4, "4-leg $2+"), (3, "3-leg $2+")):
        best = fallback = None
        for combo in combinations(wide, n):
            if not _ok_combo([cands[i][0] for i in combo]):
                continue
            pr = multi_price(combo)
            if pr >= MIN_MULTI:
                c = chance(combo)
                if best is None or c > best[0]:
                    best = (c, combo)
            elif fallback is None or pr > fallback[0]:
                fallback = (pr, combo)
        if best:
            out[name] = list(best[1])
        elif fallback:
            out[name] = list(fallback[1])
            out.setdefault("_notes", []).append(f"no {n}-leg combo reaches ${MIN_MULTI:.2f}, showing the longest-priced one")

    # Long shot: 8-12 legs, estimated price $50+, highest chance of landing
    long_idx = _long_shot(cands, every)
    if long_idx:
        out["Long shot $50"] = long_idx

    self_check = out.setdefault("_notes", [])
    if not all_worth:
        self_check.append("3-leg value includes legs without a confirmed worth-it price")
    return out


def _long_shot(cands, every) -> list[int] | None:
    """Highest-chance 8-12 leg multi whose estimated price reaches LONG_TARGET.

    Too many legs to try every combination, so: rank legs by log(chance) + lam * log(price)
    for a sweep of lam, take the top n that fit together, keep the best that reaches the
    target, then try single swaps to improve it."""
    legs = []
    for i in every:
        price, kind = price_of(*cands[i])
        if (price is not None and price < LONG_MIN_PRICE) or (price is None and cands[i][1].model_prob > 0.86):
            continue
        legs.append(i)
    if len(legs) < min(LONG_LEGS):
        return None
    lp = {i: math.log(max(cands[i][1].model_prob, 1e-6)) for i in legs}
    lq = {i: math.log(_assumed_price(*cands[i])) for i in legs}

    def fits(combo):
        return _ok_combo([cands[i][0] for i in combo])

    def chance(combo):
        return multi_chance([cands[i][0] for i in combo], [cands[i][1].model_prob for i in combo])

    def price(combo):
        return math.exp(sum(lq[i] for i in combo))

    best = None
    for n in LONG_LEGS:
        for step in range(0, 81):
            lam = step * 0.05
            order = sorted(legs, key=lambda i: -(lp[i] + lam * lq[i]))
            combo = []
            for i in order:
                if fits(combo + [i]):
                    combo.append(i)
                if len(combo) == n:
                    break
            if len(combo) < n or price(combo) < LONG_TARGET:
                continue
            c = chance(combo)
            if best is None or c > best[0]:
                best = (c, combo)
    if best is None:
        return None
    c, combo = best
    improved = True
    while improved:  # swap one leg at a time while it helps
        improved = False
        for out_i in list(combo):
            for in_i in legs:
                if in_i in combo:
                    continue
                trial = [x for x in combo if x != out_i] + [in_i]
                if price(trial) >= LONG_TARGET and fits(trial):
                    tc = chance(trial)
                    if tc > c + 1e-9:
                        c, combo, improved = tc, trial, True
                        break
            if improved:
                break
    return sorted(combo, key=lambda i: -cands[i][1].model_prob)

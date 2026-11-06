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
LONG_MIN_LEGS = 8         # long multi: at least this many legs
LONG_MAX_LEGS = 12        # extra legs past 8 only when each is worth it on its own
LONG_MIN_PROB = 0.80      # every long-multi leg at least this likely
LONG_MIN_PRICE = 1.15     # and estimated to pay at least this
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
    margin = leg.extra.get("margin", BOOKIE_MARGIN)
    if kind == "odds":
        return price
    if kind == "fair":
        return price * margin
    return margin / res.model_prob


def choose(cands: list[tuple[Leg, LegResult]]) -> dict[str, list[int]]:
    """Returns {multi name: [indexes into cands]}"""
    every = [i for i, (l, r) in enumerate(cands) if eligible(l, r)]
    every.sort(key=lambda i: -cands[i][1].model_prob)
    every = [i for i in every if not cands[i][0].extra.get("long_only")]  # safe lines / match winners: long multi only
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

    # Long multi: 8+ legs, each 80%+ likely and paying $1.15+, best value first
    long_idx, note = _long_multi(cands)
    if long_idx:
        out["Long multi 8+"] = long_idx
    if note:
        out.setdefault("_notes", []).append(note)

    self_check = out.setdefault("_notes", [])
    if not all_worth:
        self_check.append("3-leg value includes legs without a confirmed worth-it price")
    return out



def long_ok(leg: Leg, res: LegResult) -> bool:
    return (not leg.extra.get("risky") and res.model_prob >= LONG_MIN_PROB
            and _assumed_price(leg, res) >= LONG_MIN_PRICE)


def _long_multi(cands) -> tuple[list[int] | None, str]:
    """Every leg 80%+ likely and an estimated $1.15+. Ranked by value (chance x price),
    take the best 8 that fit together, then keep adding (up to 12) only legs that are
    worth it on their own (chance x price above 1)."""
    ok = [i for i, (l, r) in enumerate(cands) if long_ok(l, r)]
    value = {i: cands[i][1].model_prob * _assumed_price(*cands[i]) for i in ok}
    ok.sort(key=lambda i: (-value[i], -cands[i][1].model_prob))
    combo: list[int] = []
    for i in ok:
        if len(combo) >= LONG_MAX_LEGS:
            break
        if len(combo) >= LONG_MIN_LEGS and value[i] <= 1.0:
            break
        if _ok_combo([cands[j][0] for j in combo + [i]]):
            combo.append(i)
    if len(combo) < 5:
        return None, f"only {len(combo)} legs are 80%+ likely at $1.15+, not enough for a long multi"
    note = "" if len(combo) >= LONG_MIN_LEGS else f"only {len(combo)} legs are 80%+ likely at $1.15+, so the long multi is short"
    return sorted(combo, key=lambda i: -cands[i][1].model_prob), note

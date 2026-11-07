"""Choosing the three multis from the candidate legs."""
from __future__ import annotations

import math
from itertools import combinations

from .model import Leg, LegResult, multi_chance

MIN_PRICE = 1.20          # Sam rarely takes a leg shorter than this
MAX_UNPRICED_MODEL = 0.83 # with no price known, a leg this likely would be priced under $1.20
BOOKIE_MARGIN = 0.92      # assumed app price = this x fair, when no price is known
POOL = 16                 # legs considered when searching combinations
MIN_MULTI = 2.00          # the "$2+" multis must be estimated to pay at least this
LONG_MIN_LEGS = 8         # long multi: at least this many legs
LONG_MAX_LEGS = 12        # extra legs past 8 only when each is worth it on its own
LONG_MIN_PROB = 0.80      # every long-multi leg at least this likely
LONG_MIN_PRICE = 1.15     # and estimated to pay at least this
WIDE_POOL = 20            # $2+ multis search the top legs plus up to this many step-up (next line up) legs
MID_TARGET = 8.00         # mid-range multi: estimated price at least this
MID_LEGS = (6, 7)         # try 6 legs first, 7 only if 6 can't reach the target
MID_POOL = 22             # legs searched for the mid-range multi (most likely first)
SHOT_TARGET = 50.00       # long shot: keep adding the most likely legs until the price passes this
SHOT_MAX_LEGS = 15        # but never more than this many legs
SHOT_MIN_PRICE = 20.00    # if the slate can't get the long shot this far, leave it out
SHOT_MIN_PROB = 0.40      # no long-shot / mid-range leg less likely than this
WIDE_STEP_MIN = 0.55      # step-up legs used by the $2+ multis must be at least this likely
PER_GAME_MAX = 8          # one-per-game multi: at most this many games


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


def app_price(legs: list[Leg]) -> tuple[str | None, float | None]:
    """A multi is placed in one app: the app whose prices for every leg multiply to the most."""
    books = [l.extra.get("books") or {} for l in legs]
    if not legs or not all(books):
        return None, None
    common = set(books[0]).intersection(*books[1:])
    if not common:
        return None, None
    by_app = {bk: math.prod(b[bk] for b in books) for bk in common}
    bk = max(by_app, key=by_app.get)
    return bk, by_app[bk]


def combo_price(cands, combo) -> float:
    """The whole multi's price in the best app when one app has every leg, else the legs' prices multiplied."""
    _, pr = app_price([cands[i][0] for i in combo])
    if pr:
        return pr
    price = 1.0
    for i in combo:
        price *= _assumed_price(*cands[i])
    return price


def choose(cands: list[tuple[Leg, LegResult]]) -> dict[str, list[int]]:
    """Returns {multi name: [indexes into cands]}"""
    every = [i for i, (l, r) in enumerate(cands) if eligible(l, r)]
    every.sort(key=lambda i: -cands[i][1].model_prob)
    every_any = list(every)  # includes match winners / safe lines, for the long shot
    every = [i for i in every if not cands[i][0].extra.get("long_only")]  # safe lines / match winners: long multi only
    idx = [i for i in every if not cands[i][0].extra.get("step_up")]  # step-up legs only feed the $2+ multis
    pool = idx[:POOL]
    out: dict[str, list[int]] = {}

    def chance(combo):
        return multi_chance([cands[i][0] for i in combo], [cands[i][1].model_prob for i in combo])

    def multi_price(combo):
        return combo_price(cands, combo)

    def pays(combo):  # Sam doesn't bet multis under $2
        return multi_price(combo) >= MIN_MULTI

    def ev(combo):
        p = chance(combo)
        return p * multi_price(combo), p

    # Safest: highest chance all four land
    best = None
    for combo in combinations(pool, 4):
        if _ok_combo([cands[i][0] for i in combo]) and pays(combo):
            c = chance(combo)
            if best is None or c > best[0]:
                best = (c, combo)
    if best:
        out["Safest"] = list(best[1])
    else:
        out.setdefault("_notes", []).append("no safest 4-leg pays $2, so it's left out")

    # Best value: highest expected return, keeping a sensible chance of landing
    best = None
    for combo in combinations(pool, 4):
        if not _ok_combo([cands[i][0] for i in combo]):
            continue
        if not pays(combo):
            continue
        e, p = ev(combo)
        if p >= 0.30 and (best is None or e > best[0]):
            best = (e, combo)
    if best:
        out["Best value"] = list(best[1])
    else:
        out.setdefault("_notes", []).append("no best-value 4-leg pays $2 with a 30%+ chance, so it's left out")

    # 3-leg value: every leg must be worth it on its own (model x price > 1)
    worth = [i for i in pool if price_of(*cands[i])[0] and cands[i][1].model_prob * _assumed_price(*cands[i]) > 1]

    def best_three(search, min_p):
        best = None
        for combo in combinations(search, 3):
            if not _ok_combo([cands[i][0] for i in combo]) or not pays(combo):
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
    else:
        all_worth = True
        out.setdefault("_notes", []).append("no 3-leg value multi pays $2, so it's left out")
    # $2+ multis: highest chance all land, with the estimated Sportsbet price at least $2.00
    wide = idx[:WIDE_POOL] + [i for i in every if cands[i][0].extra.get("step_up")
                              and cands[i][1].model_prob >= WIDE_STEP_MIN][:WIDE_POOL]

    targets = {4: [("4-leg $2+", 2.0), ("4-leg $3+", 3.0), ("4-leg $4+", 4.0)], 3: [("3-leg $2+", 2.0)]}
    for n, wanted in targets.items():
        best = {name: None for name, _ in wanted}
        fallback = None
        for combo in combinations(wide, n):
            if not _ok_combo([cands[i][0] for i in combo]):
                continue
            pr = multi_price(combo)
            c = None
            for name, target in wanted:
                if pr >= target:
                    c = chance(combo) if c is None else c
                    if best[name] is None or c > best[name][0]:
                        best[name] = (c, combo)
            if fallback is None or pr > fallback[0]:
                fallback = (pr, combo)
        for name, target in wanted:
            if best[name]:
                out[name] = list(best[name][1])
            elif fallback:
                out[name] = list(fallback[1])
                out.setdefault("_notes", []).append(f"no {n}-leg combo reaches ${target:.2f}, so {name} shows the longest-priced one")

    # Long multi: 8+ legs, each 80%+ likely and paying $1.15+, best value first
    long_idx, note = _long_multi(cands)
    if long_idx:
        out["Long multi 8+"] = long_idx
    if note:
        out.setdefault("_notes", []).append(note)

    # One per game: the most likely leg from each game, so one blowout can only cost one leg
    per_game, note = _one_per_game(cands, idx, multi_price)
    if per_game:
        out["One per game"] = per_game
    if note:
        out.setdefault("_notes", []).append(note)

    # Mid-range: 6 (or 7) legs, highest chance with an estimated price of $8+
    vp = _value_pool(cands, every_any)
    by_value = sorted(vp, key=lambda i: _chance_cost(*cands[i]))
    by_chance = sorted(vp, key=lambda i: -cands[i][1].model_prob)
    mid_pool = list(dict.fromkeys(by_chance[:MID_POOL // 2] + by_value))[:MID_POOL]  # half safest, half best value
    mid, note = _target_multi(cands, mid_pool, MID_LEGS, MID_TARGET, chance, multi_price)
    if mid:
        out["Mid-range $8+"] = mid
    if note:
        out.setdefault("_notes", []).append(note)

    # Long shot: most likely legs, added one at a time until the price passes $50
    shot, note = _long_shot(cands, every_any)
    if shot:
        out["Long shot $50"] = shot
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


def _target_multi(cands, pool, sizes, target, chance, multi_price) -> tuple[list[int] | None, str]:
    """Highest chance combo of the first size in `sizes` that reaches `target`.
    Falls back to the longest-priced combo of the largest size."""
    fallback = None
    for n in sizes:
        best = None
        for combo in combinations(pool, n):
            pr = multi_price(combo)
            if pr < target:
                if n == sizes[-1] and (fallback is None or pr > fallback[0]) and _ok_combo([cands[i][0] for i in combo]):
                    fallback = (pr, combo)
                continue
            if not _ok_combo([cands[i][0] for i in combo]):
                continue
            c = chance(combo)
            if best is None or c > best[0]:
                best = (c, combo)
        if best:
            return sorted(best[1], key=lambda i: -cands[i][1].model_prob), ""
    if fallback:
        return (sorted(fallback[1], key=lambda i: -cands[i][1].model_prob),
                f"no {sizes[-1]}-leg combo reaches ${target:.2f}, so the mid-range multi shows the longest-priced one")
    return None, f"not enough legs that fit together for a {sizes[0]}-leg mid-range multi"


def _chance_cost(leg: Leg, res: LegResult) -> float:
    """Chance given up per unit of (log) price gained. Lower is better: the cheapest way to add price."""
    price = _assumed_price(leg, res)
    if price <= 1.01 or res.model_prob <= 0:
        return math.inf
    return -math.log(res.model_prob) / math.log(price)


def _value_pool(cands, ordered: list[int]) -> list[int]:
    return [i for i in ordered if cands[i][1].model_prob >= SHOT_MIN_PROB and _chance_cost(*cands[i]) < math.inf]


def _long_shot(cands, ordered: list[int]) -> tuple[list[int] | None, str]:
    """Reach SHOT_TARGET in at most SHOT_MAX_LEGS legs while giving up as little chance as possible.
    Take legs (one per player) in order of chance given up per unit of price, until the price passes
    the target; then swap each leg for a more likely unused one wherever the price still clears it."""
    pool = sorted(_value_pool(cands, ordered), key=lambda i: _chance_cost(*cands[i]))
    combo: list[int] = []
    for i in pool:
        if _price(cands, combo) >= SHOT_TARGET or len(combo) >= SHOT_MAX_LEGS:
            break
        if _ok_combo([cands[j][0] for j in combo + [i]]):
            combo.append(i)
    if _price(cands, combo) >= SHOT_TARGET:
        combo = _improve(cands, combo, pool, SHOT_TARGET, SHOT_MAX_LEGS)
    price = _price(cands, combo)
    if len(combo) < 5 or price < SHOT_MIN_PRICE:
        return None, f"not enough legs on the slate for a long shot (only about ${price:.0f})"
    note = "" if price >= SHOT_TARGET else f"the long shot only reaches about ${price:.0f} with {len(combo)} legs"
    return sorted(combo, key=lambda i: -cands[i][1].model_prob), note


def _improve(cands, combo: list[int], pool: list[int], target: float, max_legs: int) -> list[int]:
    """Local search: swap one leg for a more likely one, or one leg for two, whenever the price
    still clears the target and the chance all land goes up."""
    def chance(c):
        p = 1.0
        for i in c:
            p *= cands[i][1].model_prob
        return p

    def ok(c):
        return len(c) <= max_legs and _ok_combo([cands[j][0] for j in c]) and _price(cands, c) >= target

    best = chance(combo)
    improved = True
    while improved:
        improved = False
        unused = [i for i in pool if i not in combo]
        for pos in range(len(combo)):
            rest = combo[:pos] + combo[pos + 1:]
            options = [[a] for a in unused] + [[a, b] for k, a in enumerate(unused) for b in unused[k + 1:]]
            for add in options:
                trial = rest + add
                c = chance(trial)
                if c > best * 1.0001 and ok(trial):
                    combo, best, improved = trial, c, True
                    break
            if improved:
                break
    return combo


def _price(cands, combo) -> float:
    return combo_price(cands, combo)


def _one_per_game(cands, ordered: list[int], multi_price) -> tuple[list[int] | None, str]:
    """The most likely leg from each game (up to PER_GAME_MAX games, most likely first)."""
    seen: set[str] = set()
    combo: list[int] = []
    for i in ordered:  # already most likely first
        g = cands[i][0].game
        if g in seen or len(combo) >= PER_GAME_MAX:
            continue
        seen.add(g)
        combo.append(i)
    if len(combo) < 3:
        return None, f"only {len(combo)} game{'s' if len(combo) != 1 else ''} with legs left, not enough for one per game"
    if multi_price(combo) < MIN_MULTI:
        return None, f"the one-per-game multi only pays about ${multi_price(combo):.2f}, under $2, so it's left out"
    return combo, ""

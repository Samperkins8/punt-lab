"""Choosing the multis from the candidate legs."""
from __future__ import annotations

import math
from itertools import combinations

from .model import Leg, LegResult, multi_chance

MIN_PRICE = 1.20          # Sam rarely takes a leg shorter than this
MAX_UNPRICED_MODEL = 0.83 # with no price known, a leg this likely would be priced under $1.20
BOOKIE_MARGIN = 0.92      # assumed app price = this x fair, when no price is known
MIN_MULTI = 2.00          # Sam doesn't bet a multi paying under $2
MAIN_TARGET = 3.00        # the main bet: the likeliest 4 legs paying $3+ (the one Sam usually takes)
THREE_SHARED = 1          # 3-leg $2+ shares at most this many players with the main bet, so they're two bets, not one
LONG_MIN_LEGS = 8         # long multi: at least this many legs
LONG_MAX_LEGS = 12        # extra legs past 8 only when each is worth it on its own
LONG_MIN_PROB = 0.80      # every long-multi leg at least this likely
LONG_MIN_PRICE = 1.15     # and estimated to pay at least this
WIDE_POOL = 20            # short multis search the top legs plus up to this many step-up (next line up) legs
WIDE_STEP_MIN = 0.55      # step-up legs must be at least this likely
GAME_MAX_SHORT = 2        # 2- to 4-leg multis: at most this many legs from one game, so one bad game can't sink it
GAME_MAX_LONG = 3         # bigger multis: at most this many
THREES_MAX_SHORT = 1      # 2- to 4-leg multis: at most one threes leg. Threes in multis landed 69% against 76%
                          # predicted (78 legs to 23 Nov): a 2+ threes line rests on a handful of shots
SHORT_LEGS = 4
PICK_MIN_CHANCE = 0.25    # second pick: a 2- to 4-leg multi at least this likely to land
PICK_SHARED = 0           # ...sharing no player with the main bet, so one miss can't sink both
_caps = (GAME_MAX_SHORT, GAME_MAX_LONG)  # (short, long) for tonight; choose() loosens them on a tiny slate


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


def _ok_combo(legs: list[Leg], long: bool = False) -> bool:
    """The most legs allowed from one game follows the multi's size. Builders that add legs one
    at a time towards a bigger multi pass long=True."""
    players = [l.player for l in legs]
    if len(set(players)) != len(players):
        return False
    game_max = _caps[1] if long or len(legs) > SHORT_LEGS else _caps[0]
    per_game: dict[str, int] = {}
    for l in legs:
        per_game[l.game] = per_game.get(l.game, 0) + 1
        if per_game[l.game] > game_max:
            return False
    if not (long or len(legs) > SHORT_LEGS) and sum(l.stat == "threes" for l in legs) > THREES_MAX_SHORT:
        return False
    for a, b in combinations(legs, 2):
        if a.game != b.game:
            continue
        if a.stat == "rebounds" and b.stat == "rebounds":
            return False  # two players chasing the same boards
        if a.stat == "win" and b.stat == "win":
            return False  # both teams to win the same game
        if (a.stat == "win") != (b.stat == "win") and a.team != b.team:
            return False  # a team to win against a leg from the team it's playing: they pull against each other
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


def pick_score(p: float, price: float) -> float:
    """How good a bet is, chance and price together. When the price beats the chance it's the Kelly share
    (edge over the winnings), which favours the likelier of two equally good prices; when it doesn't,
    it's the loss per $1, so any multi worth taking ranks above every one that isn't."""
    edge = p * price - 1
    return edge / (price - 1) if edge > 0 and price > 1 else edge


def choose(cands: list[tuple[Leg, LegResult]]) -> dict[str, list[int]]:
    """Returns {multi name: [indexes into cands]}: the main bet (4-leg $3+), a 3-leg $2+ and a second pick
    built to share as few players with it as possible, and a long multi.

    Cut down from 14 multis on 24 Nov 2026. The legs were landing about as often as predicted (73% v 75%),
    but the multis landed 61 of 208 against 74 expected: they shared so many legs that one miss sank most of
    the day (0 of 10, 0 of 11, 0 of 14, then 12 of 13). Fourteen near-identical multis were really one bet
    a day, so fewer multis, deliberately spread apart."""
    global _caps
    n_games = len({l.game for l, _ in cands}) or 1
    # with one game on, everything is same-game; with two, a long multi needs four from each
    _caps = (GAME_MAX_SHORT if n_games >= 2 else SHORT_LEGS,
             max(GAME_MAX_LONG, -(-LONG_MIN_LEGS // n_games)))
    every = [i for i, (l, r) in enumerate(cands) if eligible(l, r)]
    every.sort(key=lambda i: -cands[i][1].model_prob)
    every = [i for i in every if not cands[i][0].extra.get("long_only")]  # safe lines / match winners: long multi only
    idx = [i for i in every if not cands[i][0].extra.get("step_up")]  # step-up legs only feed the $2+ multis
    wide = idx[:WIDE_POOL] + [i for i in every if cands[i][0].extra.get("step_up")
                              and cands[i][1].model_prob >= WIDE_STEP_MIN][:WIDE_POOL]
    out: dict[str, list[int]] = {}
    notes: list[str] = out.setdefault("_notes", [])

    def chance(combo):
        return multi_chance([cands[i][0] for i in combo], [cands[i][1].model_prob for i in combo])

    def players(combo):
        return {cands[i][0].player for i in combo}

    # every 2- to 4-leg combo that fits together: (size, price, combo)
    combos = [(n, combo_price(cands, c), c) for n in (2, 3, 4) for c in combinations(wide, n)
              if _ok_combo([cands[i][0] for i in c])]

    def likeliest(n, target, avoid=frozenset(), max_shared=0):
        best = None
        for size, pr, c in combos:
            if size == n and pr >= target and len(players(c) & avoid) <= max_shared:
                ch = chance(c)
                if best is None or ch > best[0]:
                    best = (ch, c)
        return list(best[1]) if best else None

    main = likeliest(4, MAIN_TARGET)
    if main:
        out["4-leg $3+"] = main
    else:
        longest = max((x for x in combos if x[0] == 4), key=lambda x: x[1], default=None)
        if longest:
            out["4-leg $3+"] = list(longest[2])
            notes.append(f"no 4-leg combo reaches ${MAIN_TARGET:.2f}, so 4-leg $3+ shows the longest-priced one")
    taken = frozenset(players(out.get("4-leg $3+", [])))

    # 3-leg $2+: the likeliest three paying $2 that shares at most THREE_SHARED players with the main bet
    for shared in range(THREE_SHARED, 4):
        three = likeliest(3, MIN_MULTI, taken, shared)
        if three:
            out["3-leg $2+"] = three
            if shared > THREE_SHARED:
                notes.append(f"3-leg $2+ shares {shared} players with 4-leg $3+: the slate is thin")
            break
    else:
        notes.append("no 3-leg combo pays $2, so it's left out")

    # Second pick: the best 2- to 4-leg bet by chance and price together that shares nobody with the main bet.
    # Combos one app prices in full come first: without a real price the edge is only a guess.
    for shared in range(PICK_SHARED, SHORT_LEGS + 1):  # on a thin slate, allow an overlap rather than no pick
        rated = []
        for size, pr, c in combos:
            if pr < MIN_MULTI or len(players(c) & taken) > shared:
                continue
            ch = chance(c)
            if ch >= PICK_MIN_CHANCE:
                real = app_price([cands[i][0] for i in c])[1] is not None
                rated.append((real, pick_score(ch, pr), ch, c))
        if rated:
            out["Second pick"] = sorted(max(rated, key=lambda r: r[:3])[3], key=lambda i: -cands[i][1].model_prob)
            break

    # Long multi: 8+ legs, each 80%+ likely and paying $1.15+, best value first
    long_idx, note = _long_multi(cands)
    if long_idx:
        out["Long multi 8+"] = long_idx
    if note:
        notes.append(note)
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
        if _ok_combo([cands[j][0] for j in combo + [i]], long=True):
            combo.append(i)
    if len(combo) < 5:
        return None, f"only {len(combo)} legs are 80%+ likely at $1.15+, not enough for a long multi"
    note = "" if len(combo) >= LONG_MIN_LEGS else f"only {len(combo)} legs are 80%+ likely at $1.15+, so the long multi is short"
    return sorted(combo, key=lambda i: -cands[i][1].model_prob), note

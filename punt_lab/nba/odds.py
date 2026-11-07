"""Optional: player prop prices from Australian bookies via The Odds API (https://the-odds-api.com).

Only runs when the ODDS_API_KEY secret is set. Requests are charged per market per game,
so the daily run first works out which games and stats today's legs need, then prices
those, busiest first, until the day's credit budget (ODDS_DAILY_CREDITS) is used.

For each stat priced we take two markets from the "au" region:
- the main over/under line (e.g. Over/Under 22.5 points) from each bookie: with the margin
  removed this is the market's view of the player (the fair price and the 60/40 blend);
- the alternate ladder (Over 9.5, 14.5, 19.5 ...): each bookie's actual price for our
  "10+", "15+", "20+" legs, so the page can show who pays best.
"""
from __future__ import annotations

import logging
import os
from statistics import median

import requests

log = logging.getLogger(__name__)
BASE = "https://api.the-odds-api.com/v4/sports/basketball_nba"
STAT_MARKET = {"points": "player_points", "rebounds": "player_rebounds",
               "assists": "player_assists", "threes": "player_threes"}
MARKET_STAT = {v: k for k, v in STAT_MARKET.items()}
MARKET_STAT.update({v + "_alternate": k for k, v in STAT_MARKET.items()})
BOOK_NAMES = {"sportsbet": "Sportsbet", "tab": "TAB", "tabtouch": "TABtouch", "ladbrokes_au": "Ladbrokes",
              "neds": "Neds", "pointsbetau": "PointsBet", "unibet": "Unibet", "betr_au": "Betr",
              "betright": "BetRight", "playup": "PlayUp", "bet365_au": "Bet365", "dabble_au": "Dabble",
              "picklebet_au": "Picklebet", "betfair_ex_au": "Betfair"}
CREDITS_PER_STAT = 2  # main line + alternate ladder


def enabled() -> bool:
    return bool(os.environ.get("ODDS_API_KEY"))


def empty() -> dict:
    return {"market": {}, "books": {}, "credits_left": None, "priced": []}


def no_vig_over_decimal(over: float, under: float) -> float:
    o, u = 1 / over, 1 / under
    return o / (o + u)


def plan(wanted: dict[tuple[str, str], dict[str, int]], budget: int) -> dict[tuple[str, str], list[str]]:
    """wanted: {(home_name, away_name): {stat: number of legs}}. Spend the budget on the
    (game, stat) pairs with the most legs first. Returns {(home, away): [stats]}."""
    pairs = sorted(((n, game, stat) for game, stats in wanted.items() for stat, n in stats.items()),
                   key=lambda x: -x[0])
    out: dict[tuple[str, str], list[str]] = {}
    spent = 0
    for n, game, stat in pairs:
        if spent + CREDITS_PER_STAT > budget:
            break
        out.setdefault(game, []).append(stat)
        spent += CREDITS_PER_STAT
    return out


def fetch(games: list[dict], wanted: dict[tuple[str, str], dict[str, int]] | None = None) -> dict:
    """Returns {'market': {(player_lower, stat): (line, p_over, source)},
                'books': {(player_lower, stat, milestone): {bookie: decimal price}},
                'credits_left': int | None, 'priced': ['NYK @ CLE points', ...]}"""
    out = empty()
    if not enabled():
        return out
    key = os.environ["ODDS_API_KEY"]
    budget = int(os.environ.get("ODDS_DAILY_CREDITS", "16"))
    if wanted is None:  # no plan given: points and rebounds for every game
        wanted = {(g["home_name"], g["away_name"]): {"points": 2, "rebounds": 1} for g in games}
    todo = plan(wanted, budget)
    if not todo:
        return out
    try:
        events = requests.get(f"{BASE}/events", params={"apiKey": key}, timeout=20).json()  # free
    except (requests.RequestException, ValueError) as e:
        log.warning("Odds API events failed: %s", e)
        return out
    abbr = {(g["home_name"], g["away_name"]): f"{g['away']} @ {g['home']}" for g in games}
    for ev in events if isinstance(events, list) else []:
        game = (ev.get("home_team"), ev.get("away_team"))
        stats = todo.get(game)
        if not stats:
            continue
        markets = ",".join(m for s in stats for m in (STAT_MARKET[s], STAT_MARKET[s] + "_alternate"))
        try:
            r = requests.get(f"{BASE}/events/{ev['id']}/odds", timeout=20, params={
                "apiKey": key, "regions": "au", "markets": markets, "oddsFormat": "decimal"})
            data = r.json()
            left = r.headers.get("x-requests-remaining")
            out["credits_left"] = int(float(left)) if left else out["credits_left"]
        except (requests.RequestException, ValueError) as e:
            log.warning("Odds API event odds failed: %s", e)
            continue
        if not isinstance(data, dict):
            log.warning("Odds API: %s", data)
            continue
        out["priced"].append(f"{abbr.get(game, ev.get('away_team'))} {'/'.join(stats)}")
        log.info("Odds API %s: %s; books %s; credits left %s", abbr.get(game), markets,
                 ", ".join(b.get("key") for b in data.get("bookmakers", [])) or "none", out["credits_left"])
        _parse(data, out)
    return out


def _parse(data: dict, out: dict) -> None:
    lines: dict[tuple[str, str], list[tuple[float, float]]] = {}
    for book in data.get("bookmakers", []):
        bk = book.get("key")
        for mk in book.get("markets", []):
            stat = MARKET_STAT.get(mk.get("key"))
            if not stat:
                continue
            main = not mk["key"].endswith("_alternate")
            sides: dict[tuple[str, float], dict[str, float]] = {}
            for oc in mk.get("outcomes", []):
                player = (oc.get("description") or "").lower()
                point, price, side = oc.get("point"), oc.get("price"), (oc.get("name") or "").lower()
                if not player or point is None or not price or price <= 1:
                    continue
                if side == "over":
                    milestone = int(float(point) + 0.5)  # "Over 19.5" pays if he gets 20+
                    slot = out["books"].setdefault((player, stat, milestone), {})
                    slot[bk] = max(slot.get(bk, 0), round(float(price), 2))
                if main:
                    sides.setdefault((player, float(point)), {})[side] = float(price)
            for (player, point), s in sides.items():
                if "over" in s and "under" in s:
                    lines.setdefault((player, stat), []).append((point, no_vig_over_decimal(s["over"], s["under"])))
    for (player, stat), vals in lines.items():
        line = median(v[0] for v in vals)
        at_line = [p for (pt, p) in vals if pt == line] or [0.5]
        out["market"][(player, stat)] = (line, median(at_line), "AU bookies")


def best(books: dict[str, float] | None) -> tuple[str | None, float | None]:
    if not books:
        return None, None
    bk = max(books, key=books.get)
    return bk, books[bk]


def typical(books: dict[str, float] | None) -> float | None:
    """The median bookie's price: what a leg pays without shopping around."""
    return round(median(books.values()), 2) if books else None

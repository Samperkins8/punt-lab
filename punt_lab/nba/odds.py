"""Optional: player prop prices from The Odds API (https://the-odds-api.com).

Only runs when the ODDS_API_KEY secret is set. Each event costs credits
(markets x regions), so the number of games priced per run is capped.

From US books we take each player's over/under line -> the market's view of
his average (used for the fair price and the 60/40 market blend).
From Sportsbet (region "au", if the API returns its props) we take actual
milestone prices: an "Over 19.5 points" price is the "20+ points" leg.
"""
from __future__ import annotations

import logging
import os
from statistics import median

import requests

from .model import no_vig_over

log = logging.getLogger(__name__)
BASE = "https://api.the-odds-api.com/v4/sports/basketball_nba"
MARKET_STAT = {"player_points": "points", "player_rebounds": "rebounds",
               "player_assists": "assists", "player_threes": "threes"}


def enabled() -> bool:
    return bool(os.environ.get("ODDS_API_KEY"))


def fetch(games: list[dict]) -> dict:
    """Returns {'market': {(player_lower, stat): (line, p_over, source)},
                'sportsbet': {(player_lower, stat, milestone): price}}"""
    out = {"market": {}, "sportsbet": {}}
    if not enabled():
        return out
    key = os.environ["ODDS_API_KEY"]
    markets = os.environ.get("ODDS_MARKETS", "player_points,player_rebounds")
    regions = os.environ.get("ODDS_REGIONS", "us,au")
    max_events = int(os.environ.get("ODDS_MAX_EVENTS", "4"))
    try:
        events = requests.get(f"{BASE}/events", params={"apiKey": key}, timeout=20).json()
    except (requests.RequestException, ValueError) as e:
        log.warning("Odds API events failed: %s", e)
        return out
    wanted = {(g["home_name"], g["away_name"]) for g in games[:max_events]}
    for ev in events if isinstance(events, list) else []:
        if (ev.get("home_team"), ev.get("away_team")) not in wanted:
            continue
        try:
            r = requests.get(f"{BASE}/events/{ev['id']}/odds", timeout=20, params={
                "apiKey": key, "regions": regions, "markets": markets, "oddsFormat": "american"})
            data = r.json()
            log.info("Odds API credits left: %s", r.headers.get("x-requests-remaining"))
        except (requests.RequestException, ValueError) as e:
            log.warning("Odds API event odds failed: %s", e)
            continue
        lines: dict[tuple[str, str], list[tuple[float, float]]] = {}
        for book in data.get("bookmakers", []):
            for mk in book.get("markets", []):
                stat = MARKET_STAT.get(mk.get("key"))
                if not stat:
                    continue
                sides: dict[tuple[str, float], dict[str, float]] = {}
                for oc in mk.get("outcomes", []):
                    player = (oc.get("description") or "").lower()
                    point, price = oc.get("point"), oc.get("price")
                    if not player or point is None or price is None:
                        continue
                    sides.setdefault((player, float(point)), {})[oc.get("name", "").lower()] = float(price)
                    if book.get("key") == "sportsbet" and oc.get("name", "").lower() == "over":
                        dec = price / 100 + 1 if price > 0 else 100 / -price + 1
                        out["sportsbet"][(player, stat, int(float(point) + 0.5))] = round(dec, 2)
                if book.get("key") == "sportsbet":
                    continue
                for (player, point), s in sides.items():
                    if "over" in s and "under" in s:
                        lines.setdefault((player, stat), []).append((point, no_vig_over(s["over"], s["under"])))
        for (player, stat), vals in lines.items():
            pts = [v[0] for v in vals]
            line = median(pts)
            at_line = [p for (pt, p) in vals if pt == line] or [0.5]
            out["market"][(player, stat)] = (line, median(at_line), "The Odds API")
    return out

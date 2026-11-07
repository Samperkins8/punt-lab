"""Optional: NBA player prop prices from OddsPapi (https://oddspapi.io), free tier.

Only runs when the ODDSPAPI_KEY secret is set. One request per bookie returns every NBA
game on the board, every player line (Over and Under), so the whole slate costs one
request per bookie: about 4 a day, ~120 a month against the free 250.

Returns the same shape as odds.fetch, so the two sources merge:
  {'market': {(player, stat): [(line, p_over), ...]}, 'books': {(player, stat, milestone): {bookie: price}}}
"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

log = logging.getLogger(__name__)
BASE = "https://api.oddspapi.io/v4"
NBA = 132
BASKETBALL = 11
# OddsPapi slug -> the key used everywhere else (matches The Odds API's keys where they overlap)
BOOKS = {"pointsbet.com.au": "pointsbetau", "neds.com.au": "neds", "unibet.com.au": "unibet", "tabtouch": "tabtouch"}
MARKET_NAMES = {"Over Under Player Points (incl. overtime)": "points",
                "Over Under Player Rebounds (incl. overtime)": "rebounds",
                "Over Under Player Assists (incl. overtime)": "assists",
                "Over Under Player 3 Point FG (incl. overtime)": "threes"}
CACHE = Path(__file__).resolve().parent / "cache" / "oddspapi_markets.json"
CACHE_DAYS = 7


def enabled() -> bool:
    return bool(os.environ.get("ODDSPAPI_KEY"))


def _get(path: str, **params):
    params["apiKey"] = os.environ["ODDSPAPI_KEY"]
    try:
        r = requests.get(f"{BASE}{path}", params=params, timeout=90)
        if r.status_code != 200:
            log.warning("OddsPapi %s: HTTP %s %s", path, r.status_code, r.text[:200])
            return None
        return r.json()
    except (requests.RequestException, ValueError) as e:
        log.warning("OddsPapi %s failed: %s", path, e)
        return None


def market_defs() -> dict[int, dict]:
    """{marketId: {stat, handicap, outcomes: {outcomeId: 'over'|'under'}}} for full-game player points,
    rebounds, assists and threes. Cached in the repo for a week (one request to refresh)."""
    try:
        cached = json.loads(CACHE.read_text())
        if time.time() - cached["fetched"] < CACHE_DAYS * 86400:
            return {int(k): v for k, v in cached["markets"].items()}
    except (OSError, ValueError, KeyError):
        pass
    data = _get("/markets")
    if not isinstance(data, list):
        return {}
    defs = {}
    for m in data:
        stat = MARKET_NAMES.get(m.get("marketName"))
        if m.get("sportId") == BASKETBALL and m.get("playerProp") and stat and m.get("handicap") is not None:
            defs[int(m["marketId"])] = {"stat": stat, "handicap": float(m["handicap"]),
                                        "outcomes": {str(o["outcomeId"]): o["outcomeName"].lower() for o in m.get("outcomes", [])}}
    CACHE.parent.mkdir(exist_ok=True)
    CACHE.write_text(json.dumps({"fetched": time.time(), "markets": defs}, separators=(",", ":")))
    return defs


def player_key(name: str) -> str:
    """'Jackson Jr., Jaren' -> 'jaren jackson jr.'  (ESPN order, lower case)"""
    name = (name or "").strip()
    if "," in name:
        last, first = name.split(",", 1)
        name = f"{first.strip()} {last.strip()}"
    return name.lower()


def fetch(starts: list[datetime]) -> dict:
    """Prices for games starting within an hour of any of `starts` (today's tip-off times)."""
    out = {"market": {}, "books": {}, "requests": 0, "bookies": []}
    if not enabled() or not starts:
        return out
    defs = market_defs()
    if not defs:
        return out
    lo = min(starts).astimezone(timezone.utc) - timedelta(hours=1)
    hi = max(starts).astimezone(timezone.utc) + timedelta(hours=1)
    for slug, key in BOOKS.items():
        data = _get("/odds-by-tournaments", tournamentIds=NBA, bookmaker=slug, oddsFormat="decimal")
        out["requests"] += 1
        fixtures = data if isinstance(data, list) else []
        n = 0
        for f in fixtures:
            try:
                start = datetime.fromisoformat(f["startTime"].replace("Z", "+00:00"))
            except (KeyError, ValueError):
                continue
            if not lo <= start <= hi:
                continue
            bo = (f.get("bookmakerOdds") or {}).get(slug) or {}
            if bo.get("suspended"):
                continue
            n += _parse(bo.get("markets") or {}, defs, key, out)
        if n:
            out["bookies"].append(key)
        log.info("OddsPapi %s: %d player prices", slug, n)
        time.sleep(1.1)  # endpoint cooldown
    return out


def _parse(markets: dict, defs: dict, book: str, out: dict) -> int:
    n = 0
    sides: dict[tuple[str, str, float], dict] = {}
    for mid, m in markets.items():
        d = defs.get(int(mid)) if str(mid).isdigit() else None
        if not d or not m.get("marketActive", True):
            continue
        for oid, o in (m.get("outcomes") or {}).items():
            side = d["outcomes"].get(str(oid))
            for pl in (o.get("players") or {}).values():
                price = pl.get("price")
                if not pl.get("active", True) or not price or price <= 1:
                    continue
                player = player_key(pl.get("playerName"))
                if not player:
                    continue
                s = sides.setdefault((player, d["stat"], d["handicap"]), {})
                s[side] = float(price)
                s["main"] = s.get("main") or bool(pl.get("mainLine"))
                if side == "over":
                    slot = out["books"].setdefault((player, d["stat"], int(d["handicap"] + 0.5)), {})
                    slot[book] = max(slot.get(book, 0), round(float(price), 2))
                    n += 1
    # market view: this bookie's main line (or the line closest to even money), margin removed
    by_player: dict[tuple[str, str], list] = {}
    for (player, stat, line), s in sides.items():
        if "over" in s and "under" in s:
            o, u = 1 / s["over"], 1 / s["under"]
            by_player.setdefault((player, stat), []).append((s["main"], abs(o / (o + u) - 0.5), line, o / (o + u)))
    for k, rows in by_player.items():
        main = [r for r in rows if r[0]] or sorted(rows, key=lambda r: r[1])[:1]
        out["market"].setdefault(k, []).append((main[0][2], main[0][3]))
    return n

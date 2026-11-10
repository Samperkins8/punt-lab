"""Free ESPN public JSON feeds: schedule, box scores, injuries, player game logs.

These endpoints are unofficial and unauthenticated. Every parser here is
defensive: a missing field gives None/empty rather than a crash, and the
caller logs what it skipped.
"""
from __future__ import annotations

import logging
import re
import time
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import requests

log = logging.getLogger(__name__)
SITE = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
WEB = "https://site.web.api.espn.com/apis/common/v3/sports/basketball/nba"
MELB = ZoneInfo("Australia/Melbourne")
HEADERS = {"User-Agent": "punt-lab/1.0 (personal stats project)"}

_session = requests.Session()
_session.headers.update(HEADERS)
_cache: dict[str, dict] = {}


def get(url: str, params: dict | None = None, retries: int = 3) -> dict:
    key = url + "?" + "&".join(f"{k}={v}" for k, v in sorted((params or {}).items()))
    if key in _cache:
        return _cache[key]
    last = None
    for attempt in range(retries):
        try:
            r = _session.get(url, params=params, timeout=20)
            if r.status_code == 200:
                data = r.json()
                _cache[key] = data
                time.sleep(0.15)
                return data
            last = f"HTTP {r.status_code}"
        except (requests.RequestException, ValueError) as e:
            last = str(e)
        time.sleep(1.5 * (attempt + 1))
    log.warning("GET failed %s %s: %s", url, params, last)
    return {}


# ---------- slate ----------

def slate_games(slate: date) -> list[dict]:
    """Games whose tip-off falls on `slate` in Melbourne time."""
    games = []
    seen = set()
    for us_day in (slate - timedelta(days=1), slate):
        data = get(f"{SITE}/scoreboard", {"dates": us_day.strftime("%Y%m%d")})
        for ev in data.get("events", []):
            if ev.get("id") in seen:
                continue
            try:
                start = datetime.fromisoformat(ev["date"].replace("Z", "+00:00")).astimezone(MELB)
            except (KeyError, ValueError):
                continue
            if start.date() != slate:
                continue
            seen.add(ev["id"])
            comp = (ev.get("competitions") or [{}])[0]
            teams = {c.get("homeAway"): c.get("team", {}) for c in comp.get("competitors", [])}
            home, away = teams.get("home", {}), teams.get("away", {})
            odds = (comp.get("odds") or [{}])[0]
            spread_fav, spread = parse_spread(odds.get("details"))
            games.append({
                "event_id": ev["id"],
                "start": start,
                "home": home.get("abbreviation"), "home_id": home.get("id"), "home_name": home.get("displayName"),
                "away": away.get("abbreviation"), "away_id": away.get("id"), "away_name": away.get("displayName"),
                "favourite": spread_fav,
                "spread": spread,
                "total": _num(odds.get("overUnder")),
                "state": comp.get("status", {}).get("type", {}).get("state"),  # pre / in / post
            })
    games.sort(key=lambda g: g["start"])
    return games


def parse_spread(details: str | None) -> tuple[str | None, float | None]:
    # e.g. "LAL -3.5" or "EVEN"
    if not details or " " not in details:
        return None, None
    abbr, val = details.rsplit(" ", 1)
    try:
        return abbr.strip(), float(val)
    except ValueError:
        return None, None


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# ---------- team games and box scores ----------

def team_completed_games(team_id: str) -> list[dict]:
    data = get(f"{SITE}/teams/{team_id}/schedule")
    out = []
    for ev in data.get("events", []):
        comp = (ev.get("competitions") or [{}])[0]
        if not comp.get("status", {}).get("type", {}).get("completed"):
            continue
        st = (ev.get("seasonType") or {}).get("type") or (ev.get("seasonType") or {}).get("id")
        if st and str(st) not in ("2", "regular"):  # regular season only (skip preseason)
            continue
        home = None
        for c in comp.get("competitors", []):
            if str((c.get("team") or {}).get("id")) == str(team_id):
                home = c.get("homeAway") == "home"
        out.append({"event_id": ev["id"], "date": ev.get("date"), "home": home})
    out.sort(key=lambda g: g["date"] or "")
    return out


def box_score(event_id: str) -> dict:
    """Returns {team_abbr: [{id, name, pos, starter, dnp, min, pts, reb, ast, threes}]}"""
    data = get(f"{SITE}/summary", {"event": event_id})
    out: dict[str, list[dict]] = {}
    for team_block in (data.get("boxscore") or {}).get("players", []):
        abbr = (team_block.get("team") or {}).get("abbreviation")
        stats_block = (team_block.get("statistics") or [{}])[0]
        labels = [str(x).upper() for x in stats_block.get("labels") or stats_block.get("names") or []]
        idx = {lab: i for i, lab in enumerate(labels)}
        rows = []
        for a in stats_block.get("athletes", []):
            ath = a.get("athlete") or {}
            stats = a.get("stats") or []
            dnp = bool(a.get("didNotPlay")) or not stats
            def s(label):
                i = idx.get(label)
                return stats[i] if i is not None and i < len(stats) else None
            row = {
                "id": str(ath.get("id")), "name": ath.get("displayName"),
                "pos": (ath.get("position") or {}).get("abbreviation"),
                "starter": bool(a.get("starter")), "dnp": dnp,
                "min": _minutes(s("MIN")), "pts": _num(s("PTS")), "reb": _num(s("REB")),
                "ast": _num(s("AST")), "threes": _made(s("3PT")),
            }
            if row["min"] in (None, 0):
                row["dnp"] = True
            rows.append(row)
        if abbr:
            out[abbr] = rows
    return out


def _minutes(v):
    if v is None:
        return None
    try:
        return float(str(v).split(":")[0])
    except ValueError:
        return None


def _made(v):
    if v is None:
        return None
    try:
        return float(str(v).split("-")[0])
    except ValueError:
        return None


# ---------- injuries ----------

def injuries() -> dict[str, dict]:
    """{player name lower: {status, comment, team}}"""
    data = get(f"{SITE}/injuries")
    out = {}
    for team in data.get("injuries", []):
        tname = team.get("displayName")
        for inj in team.get("injuries", []):
            ath = inj.get("athlete") or {}
            name = ath.get("displayName")
            if not name:
                continue
            out[name.lower()] = {
                "status": (inj.get("status") or "").strip(),
                "comment": inj.get("shortComment") or inj.get("longComment") or "",
                "team": tname,
            }
    return out


# ---------- last season (prior and head-to-head) ----------

STAT_LABEL = {"points": "PTS", "rebounds": "REB", "assists": "AST", "threes": "3PT", "minutes": "MIN"}


def team_roster(team_id: str) -> list[dict]:
    """[{id, name, pos}] for a team's current roster."""
    data = get(f"{SITE}/teams/{team_id}/roster")
    out = []
    for a in data.get("athletes", []):
        items = a.get("items") if isinstance(a, dict) and "items" in a else [a]
        for x in items:
            if x.get("id"):
                out.append({"id": str(x["id"]), "name": x.get("displayName"),
                            "pos": (x.get("position") or {}).get("abbreviation")})
    return out


def _margin(info: dict, home: bool | None) -> float | None:
    """His team's winning margin in a game (negative for a loss), from whichever score fields ESPN gives."""
    hs, as_ = _num(info.get("homeTeamScore")), _num(info.get("awayTeamScore"))
    if hs is not None and as_ is not None and home is not None and (hs or as_):
        return hs - as_ if home else as_ - hs
    nums = re.findall(r"\d+", str(info.get("score") or ""))
    result = str(info.get("gameResult") or "").upper()[:1]
    if len(nums) >= 2 and result in ("W", "L"):
        gap = abs(float(nums[0]) - float(nums[1]))
        return gap if result == "W" else -gap
    return None


def last_season_log(athlete_id: str, season_end_year: int) -> dict:
    """{'games': [{opp, team, pts, reb, ast, threes, min, date, home, margin}], 'gp': int}"""
    data = get(f"{WEB}/athletes/{athlete_id}/gamelog", {"season": season_end_year})
    labels = [str(x).upper() for x in data.get("labels") or []]
    idx = {lab: i for i, lab in enumerate(labels)}
    meta = data.get("events") or {}
    games = []
    for stype in data.get("seasonTypes", []):
        name = (stype.get("displayName") or "").lower()
        if "regular" not in name:
            continue
        for cat in stype.get("categories", []):
            for ev in cat.get("events", []):
                stats = ev.get("stats") or []
                info = meta.get(str(ev.get("eventId"))) or {}
                def s(label):
                    i = idx.get(label)
                    return stats[i] if i is not None and i < len(stats) else None
                home = None if not info.get("atVs") else info.get("atVs") != "@"
                games.append({
                    "margin": _margin(info, home),
                    "opp": (info.get("opponent") or {}).get("abbreviation"),
                    "team": (info.get("team") or {}).get("abbreviation"),
                    "pts": _num(s("PTS")), "reb": _num(s("REB")), "ast": _num(s("AST")),
                    "threes": _made(s("3PT")), "min": _minutes(s("MIN")),
                    "date": info.get("gameDate"), "home": home,
                })
    return {"games": games, "gp": len(games)}

"""End-to-end run of the pipeline with ESPN replaced by fake data."""
import json
import random
from datetime import datetime

from punt_lab.nba import build, espn


def fake_world(monkeypatch, tmp_path):
    rnd = random.Random(1)
    teams = {"LAL": "13", "POR": "22", "PHI": "20", "CLE": "5"}
    roster = {t: [(f"{t}{i}", f"{t} Player{i}", pos) for i, pos in enumerate(["PG", "SG", "SF", "PF", "C", "G", "F"])] for t in teams}
    events = {t: [f"{t}-g{k}" for k in range(8)] for t in teams}

    def box(ev):
        team = ev.split("-")[0]
        out = {}
        for t in (team, "OPP"):
            rows = []
            for pid, name, pos in roster.get(t, roster["LAL"]):
                star = pid.endswith(("0", "4"))
                rows.append({"id": pid if t == team else "o" + pid, "name": name, "pos": pos, "starter": True, "dnp": False,
                             "min": 34 if star else 26, "pts": rnd.gauss(27 if star else 13, 5), "reb": rnd.gauss(10 if pos == "C" else 5, 2),
                             "ast": rnd.gauss(7 if pos == "PG" else 3, 2), "threes": max(0, round(rnd.gauss(2.5, 1)))})
            out[t] = rows
        return out

    start = datetime(2026, 11, 7, 11, 0, tzinfo=espn.MELB)
    monkeypatch.setattr(espn, "slate_games", lambda d: [
        {"event_id": "E1", "start": start, "home": "LAL", "home_id": "13", "home_name": "Los Angeles Lakers",
         "away": "POR", "away_id": "22", "away_name": "Portland Trail Blazers", "favourite": "LAL", "spread": -3.5, "total": 238.5, "state": "pre"},
        {"event_id": "E2", "start": start, "home": "CLE", "home_id": "5", "home_name": "Cleveland Cavaliers",
         "away": "PHI", "away_id": "20", "away_name": "Philadelphia 76ers", "favourite": "CLE", "spread": -1.5, "total": 234.5, "state": "pre"}])
    monkeypatch.setattr(espn, "team_completed_games", lambda tid: [{"event_id": e, "date": e} for t, i in teams.items() if i == tid for e in events[t]])
    monkeypatch.setattr(espn, "box_score", box)
    monkeypatch.setattr(espn, "injuries", lambda: {"lal player2": {"status": "Out", "comment": "ankle", "team": "Lakers"}})
    monkeypatch.setattr(espn, "last_season_log", lambda pid, yr: {"games": [{"opp": "POR", "pts": 25, "reb": 6, "ast": 5, "threes": 2, "min": 33}] * 50, "gp": 50})
    monkeypatch.setattr(build, "DATA", tmp_path)
    monkeypatch.setattr(build, "ROOT", tmp_path)


def test_build_runs(monkeypatch, tmp_path):
    fake_world(monkeypatch, tmp_path)
    monkeypatch.setattr("sys.argv", ["build", "--date", "2026-11-07"])
    build.main()
    legs = json.loads((tmp_path / "legs.json").read_text())
    multis = json.loads((tmp_path / "multis.json").read_text())
    assert legs and all(0 < l["model_prob"] < 1 for l in legs)
    assert {m["multi"] for m in multis} == {"Best value", "Safest", "3-leg value", "4-leg $2+", "3-leg $2+", "4-leg $3+", "4-leg $4+"}
    assert not any(l["player"] == "LAL Player2" for l in legs)  # injured player excluded
    assert all(l["model_prob"] <= 0.83 or (l["fair_price"] or l["odds"] or 0) >= 1.2 for l in legs)

    # next morning: grading the previous slate
    monkeypatch.setattr("sys.argv", ["build", "--date", "2026-11-08"])
    build.main()
    results = json.loads((tmp_path / "results.json").read_text())
    assert results and results[0]["slate_date"] == "2026-11-07"

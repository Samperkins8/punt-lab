"""End-to-end run of the pipeline with ESPN replaced by fake data."""
import json
import random
from datetime import datetime, timedelta

from punt_lab.nba import build, espn


def fake_world(monkeypatch, tmp_path, spread=-3.5):
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
         "away": "POR", "away_id": "22", "away_name": "Portland Trail Blazers", "favourite": "LAL", "spread": spread, "total": 238.5, "state": "pre"},
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
    assert {"4-leg $2+", "3-leg $2+", "4-leg $3+", "4-leg $4+"} <= {m["multi"] for m in multis}
    assert not any(l["player"] == "LAL Player2" for l in legs)  # injured player excluded
    assert all(l["model_prob"] <= 0.83 or (l["fair_price"] or l["odds"] or 0) >= 1.2 for l in legs)
    # the promo builder's data: every rated leg, game start times, and how same-game legs move together
    pool = json.loads((tmp_path / "pool.json").read_text())
    assert len(pool) >= len([l for l in legs if l["stat"] != "win"]) and {l["leg_id"] for l in legs if l["stat"] != "win"} <= {p["leg_id"] for p in pool}
    assert all(g["start"].startswith("2026-11-07T11:00") for g in json.loads((tmp_path / "slate.json").read_text()))
    assert set(json.loads((tmp_path / "report.json").read_text())["links"]) == {"SAME_GAME_REBOUNDS", "SAME_TEAM_POINTS", "SAME_GAME_OTHER", "WIN_VS_OPPONENT"}

    # next morning: grading the previous slate
    monkeypatch.setattr("sys.argv", ["build", "--date", "2026-11-08"])
    build.main()
    results = json.loads((tmp_path / "results.json").read_text())
    assert results and results[0]["slate_date"] == "2026-11-07"


def test_late_check_marks_outs_and_keeps_morning_multis(monkeypatch, tmp_path):
    from punt_lab.nba import late
    fake_world(monkeypatch, tmp_path)
    monkeypatch.setattr("sys.argv", ["build", "--date", "2026-11-07"])
    build.main()
    multis_before = json.loads((tmp_path / "multis.json").read_text())
    legs = json.loads((tmp_path / "legs.json").read_text())
    in_multi = next(l for l in legs if any(l["leg_id"] in m["leg_ids"] for m in multis_before) and l["stat"] != "win")
    # 45 minutes before that leg's game, the player is ruled out
    games = espn.slate_games(None)
    game = next(g for g in games if f"{g['away']} @ {g['home']}" == in_multi["game"])
    now = game["start"] - timedelta(minutes=45)
    monkeypatch.setattr(espn, "injuries", lambda: {
        "lal player2": {"status": "Out", "comment": "ankle", "team": "Los Angeles Lakers"},
        in_multi["player"].lower(): {"status": "Out", "comment": "late scratch", "team": "x"}})
    assert late.run(build.date(2026, 11, 7), now)
    legs_after = {l["leg_id"]: l for l in json.loads((tmp_path / "legs.json").read_text())}
    multis_after = json.loads((tmp_path / "multis.json").read_text())
    assert legs_after[in_multi["leg_id"]]["late_out"] == "Out"
    assert [m["leg_ids"] for m in multis_after] == [m["leg_ids"] for m in multis_before]  # morning picks unchanged
    hit = [m for m in multis_after if in_multi["leg_id"] in m["leg_ids"]]
    assert hit and all("late_warning" in m and in_multi["player"] in m["late_warning"] for m in hit)
    report = json.loads((tmp_path / "report.json").read_text())
    assert in_multi["game"] in report["late_checked"]
    # checked once only
    assert not late.run(build.date(2026, 11, 7), now + timedelta(minutes=10))


def test_underdog_scorers_are_marked_down(monkeypatch, tmp_path):
    """An 8.5-point line: the underdog's scorers are predicted lower and carry a blowout note; the favourite's don't."""
    def run(spread, sub):
        out = tmp_path / sub
        out.mkdir()
        fake_world(monkeypatch, out, spread=spread)
        monkeypatch.setattr("sys.argv", ["build", "--date", "2026-11-07"])
        build.main()
        return {l["leg_id"]: l for l in json.loads((out / "legs.json").read_text())}

    close, wide = run(-1.5, "close"), run(-8.5, "wide")
    dogs = [l for l in wide.values() if l["team"] == "POR"]
    favs = [l for l in wide.values() if l["team"] == "LAL"]
    assert dogs and favs
    for l in dogs:
        assert l["underdog"] and "blowout risk" in l["note"]      # starters sit first when a game gets away
        if l["stat"] == "rebounds":
            assert l["side_factor"] == 1                           # rebounds don't follow the score
        else:
            assert l["side_factor"] < 1 and "underdogs" in l["why"]
    for l in favs:
        assert not l["underdog"] and "blowout risk" not in l["note"] and l["side_factor"] >= 1

    def means(legs, team):
        return {(l["player"], l["stat"]): l["proj_mean"] for l in legs.values() if l["team"] == team}
    lower = [(k, v) for k, v in means(wide, "POR").items() if k in means(close, "POR")]
    assert lower and all(v < means(close, "POR")[k] for k, v in lower)
    higher = [(k, v) for k, v in means(wide, "LAL").items() if k in means(close, "LAL") and k[1] != "rebounds"]
    assert higher and all(v >= means(close, "LAL")[k] for k, v in higher)   # the favourite's scorers aren't cut


def test_lineup_check_flags_a_usual_starter_on_the_bench(monkeypatch, tmp_path):
    from punt_lab.nba import late
    fake_world(monkeypatch, tmp_path)
    monkeypatch.setattr("sys.argv", ["build", "--date", "2026-11-07"])
    build.main()
    legs = json.loads((tmp_path / "legs.json").read_text())
    multis = json.loads((tmp_path / "multis.json").read_text())
    target = next(l for l in legs if l["stat"] != "win" and l["usual_starter"] and any(l["leg_id"] in m["leg_ids"] for m in multis))
    game = next(g for g in espn.slate_games(None) if f"{g['away']} @ {g['home']}" == target["game"])
    ids = {t: [str(l["athlete_id"]) for l in legs if l.get("team") == t and l["stat"] != "win"] for t in (game["home"], game["away"])}

    def five(team, drop=None):
        keep = [i for i in dict.fromkeys(ids[team]) if i != drop][:5]
        return set(keep + [f"{team}-filler{k}" for k in range(5 - len(keep))])

    now = game["start"] - timedelta(minutes=20)
    # line-ups not posted yet: nothing happens, and the game is tried again next run
    monkeypatch.setattr(espn, "starters", lambda ev, teams: {})
    assert not late.check_lineups(json.loads((tmp_path / "report.json").read_text()), [game], now)
    # posted, with the target on the bench
    posted = {t: five(t, drop=str(target["athlete_id"])) for t in ids}
    monkeypatch.setattr(espn, "starters", lambda ev, teams: posted if ev == game["event_id"] else {})
    report = json.loads((tmp_path / "report.json").read_text())
    assert late.check_lineups(report, [game], now)
    after = {l["leg_id"]: l for l in json.loads((tmp_path / "legs.json").read_text())}
    assert after[target["leg_id"]]["late_bench"] and "starting five" in after[target["leg_id"]]["note"]
    started = [l for l in after.values() if l["game"] == target["game"] and str(l.get("athlete_id")) in posted.get(l.get("team"), ())]
    assert started and not any(l.get("late_bench") for l in started)
    hit = [m for m in json.loads((tmp_path / "multis.json").read_text()) if target["leg_id"] in m["leg_ids"]]
    assert hit and all(target["player"] in m["late_warning"] for m in hit)
    assert target["game"] in report["lineup_checked"]
    assert not late.check_lineups(report, [game], now)     # once per game
    assert not late.check_lineups({}, [game], game["start"] - timedelta(minutes=50))   # too early


def test_role_change_and_starters():
    starter, bench = [True] * 8, [False] * 8
    note, mins = build.role_change(starter + [False, False], [34] * 8 + [22, 20])
    assert "off the bench" in note and mins == 21
    note, mins = build.role_change(bench + [True, True], [18] * 8 + [30, 32])
    assert "started" in note and mins == 31
    assert build.role_change(starter + [True, False], [34] * 10) is None     # one game isn't a change
    assert build.role_change([True, False, False], [30, 20, 20]) is None     # too few games to know his usual role
    row = lambda i, s: {"athlete": {"id": str(i), "displayName": f"P{i}"}, "starter": s, "stats": []}
    block = lambda abbr, n: {"team": {"abbreviation": abbr}, "statistics": [{"labels": ["MIN", "PTS"], "athletes": [row(f"{abbr}{i}", i < n) for i in range(8)]}]}
    monkeypatch_get = espn.get
    try:
        espn.get = lambda url, params=None: {"boxscore": {"players": [block("LAL", 5), block("POR", 5)]}}
        assert espn.starters("e", ("LAL", "POR")) == {"LAL": {f"LAL{i}" for i in range(5)}, "POR": {f"POR{i}" for i in range(5)}}
        espn.get = lambda url, params=None: {"boxscore": {"players": [block("LAL", 5), block("POR", 0)]}}
        assert espn.starters("e", ("LAL", "POR")) == {}      # one team's five isn't in yet
    finally:
        espn.get = monkeypatch_get

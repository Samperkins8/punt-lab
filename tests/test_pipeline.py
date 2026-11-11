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


def test_rechecks_keep_the_mornings_prices(monkeypatch, tmp_path):
    """The pre-tip and late re-checks must still know every price the morning fetched, not just the
    prices of the legs the morning happened to list (they used to swap priced legs for unpriced ones)."""
    from punt_lab.nba import late, odds, oddspapi
    fake_world(monkeypatch, tmp_path)
    names = [f"{t} Player{i}".lower() for t in ("LAL", "POR", "PHI", "CLE") for i in range(7)]

    def feed(starts, books=None):
        out = {"market": {}, "books": {}, "requests": 1, "bookies": ["neds"]}
        for n in names:
            out["market"][(n, "points")] = [(19.5, 0.5)]
            for line, pr in ((10, 1.08), (15, 1.3), (20, 1.9), (25, 3.2), (30, 6.0)):
                out["books"][(n, "points", line)] = {"neds": pr}
        return out

    monkeypatch.setattr(odds, "enabled", lambda: True)
    monkeypatch.setattr(odds, "fetch", lambda games, wanted=None: odds.empty())
    monkeypatch.setattr(odds, "fetch_h2h", lambda games: [])
    monkeypatch.setattr(oddspapi, "enabled", lambda: True)
    monkeypatch.setattr(oddspapi, "fetch", feed)
    monkeypatch.setattr("sys.argv", ["build", "--date", "2026-11-07"])
    build.main()
    morning = json.loads((tmp_path / "legs.json").read_text())
    kept = build.load_prices(build.date(2026, 11, 7))
    assert len(kept["books"]) == len(names) * 5 and len(kept["market"]) == len(names)
    assert len(kept["books"]) > len(late.restore_prices(morning)["books"])     # more than the listed legs carry
    assert build.load_prices(build.date(2026, 11, 8)) is None                  # yesterday's prices are never reused

    # the feeds are not called again; a late scratch reshuffles the legs
    monkeypatch.setattr(oddspapi, "fetch", lambda *a, **k: (_ for _ in ()).throw(AssertionError("no refetch")))
    star = next(l for l in morning if l["stat"] == "points")
    monkeypatch.setattr(espn, "injuries", lambda: {star["player"].lower(): {"status": "Out", "comment": "rest", "team": "x"}})
    monkeypatch.setattr("sys.argv", ["build", "--date", "2026-11-07", "--mode", "pretip"])
    build.main()
    after = json.loads((tmp_path / "legs.json").read_text())
    points = [l for l in after if l["stat"] == "points"]
    assert points and all(l["books"] == {"neds": l["best_price"]} and l["market_line"] == 19.5 for l in points)
    assert {l["leg_id"] for l in points} != {l["leg_id"] for l in morning if l["stat"] == "points"}   # the list did change

    # merging a refresh onto kept prices works (the kept bookie line counts as one bookie)
    merged = odds.finalise(odds.merge(kept, {"market": {(names[0], "points"): [(21.5, 0.52)]}, "books": {(names[0], "points", 20): {"neds": 2.0}}}))
    assert merged["market"][(names[0], "points")][0] in (19.5, 21.5) and merged["books"][(names[0], "points", 20)]["neds"] == 2.0


def test_live_grading_marks_multis_as_games_finish(monkeypatch, tmp_path):
    fake_world(monkeypatch, tmp_path)
    monkeypatch.setattr("sys.argv", ["build", "--date", "2026-11-07"])
    build.main()
    legs = json.loads((tmp_path / "legs.json").read_text())
    multis = json.loads((tmp_path / "multis.json").read_text())
    games = espn.slate_games(None)
    now = games[0]["start"] + timedelta(hours=6)
    assert not build.grade_live(now)  # nothing final yet
    # every leg gets exactly what it needed, then both games finish
    by_ath = {str(l.get("athlete_id")): l for l in legs if l["stat"] != "win"}
    def box(ev):
        rows = [{"id": a, "name": "", "pos": "G", "starter": True, "dnp": False, "min": 30,
                 "pts": 99, "reb": 99, "ast": 99, "threes": 99} for a in by_ath]
        return {"A": rows, "B": []}
    monkeypatch.setattr(espn, "box_score", box)
    monkeypatch.setattr(espn, "slate_games", lambda d: [{**g, "state": "post"} for g in games])
    assert build.grade_live(now)
    after = json.loads((tmp_path / "multis.json").read_text())
    prop_only = [m for m in after if all(i in {l["leg_id"] for l in legs if l["stat"] != "win"} for i in m["leg_ids"])]
    assert prop_only and all(m["landed"] is True for m in prop_only)
    assert all(l.get("landed") is True for l in json.loads((tmp_path / "legs.json").read_text()) if l["stat"] != "win")
    assert not build.grade_live(now + timedelta(minutes=30))  # each game graded once
    # a player who played well under his expected minutes is flagged as having left hurt
    rep = json.loads((tmp_path / "report.json").read_text()); rep["final_games"] = []
    (tmp_path / "report.json").write_text(json.dumps(rep))
    hurt_id = next(iter(by_ath))
    monkeypatch.setattr(espn, "box_score", lambda ev: {k: [{**r, "min": 4} if r["id"] == hurt_id else r for r in v] for k, v in box(ev).items()})
    monkeypatch.setattr(espn, "injuries", lambda: {by_ath[hurt_id]["player"].lower(): {"status": "Day-To-Day", "comment": "ankle", "team": "x"}})
    assert build.grade_live(now)
    flagged = [l for l in json.loads((tmp_path / "legs.json").read_text()) if l.get("left_early")]
    assert flagged and all(str(l["athlete_id"]) == hurt_id for l in flagged)
    assert flagged[0]["left_early"]["minutes"] == 4 and "ankle" in flagged[0]["left_early"]["injury"]
    assert all("minutes_played" in l for l in json.loads((tmp_path / "legs.json").read_text()) if l["stat"] != "win")

    # a missed leg makes its multis miss, even before the rest are decided
    first = multis[0]["leg_ids"][0]
    actual_line = next(l for l in legs if l["leg_id"] == first)
    monkeypatch.setattr(build, "leg_actual", lambda leg, boxes: (leg["line"] - 1) if leg["leg_id"] == first else leg["line"])
    rep = json.loads((tmp_path / "report.json").read_text()); rep["final_games"] = []
    (tmp_path / "report.json").write_text(json.dumps(rep))
    monkeypatch.setattr(espn, "slate_games", lambda d: [
        {**g, "state": "post" if f"{g['away']} @ {g['home']}" == actual_line["game"] else "in"} for g in games])
    assert build.grade_live(now)
    assert next(m for m in json.loads((tmp_path / "multis.json").read_text()) if m["multi"] == multis[0]["multi"])["landed"] is False


def test_minutes_flags():
    leg = {"stat": "points", "event_id": "E", "athlete_id": 7, "proj_minutes": 34}
    def boxes(mins, margin):
        return {"E": {"A": [{"id": "7", "dnp": False, "min": mins, "pts": 100 + margin}], "B": [{"id": "8", "dnp": False, "min": 30, "pts": 100}]}}
    assert build.minutes_flags(leg, boxes(33, 5))["minutes_flag"] is None
    assert build.minutes_flags(leg, boxes(10, 5))["minutes_flag"] == "injured"
    assert build.minutes_flags(leg, boxes(21, 25))["minutes_flag"] == "blowout"
    assert build.minutes_flags(leg, boxes(21, 5)) == {"minutes_played": 21, "expected_minutes": 34, "minutes_flag": "low"}
    assert build.minutes_flags({**leg, "stat": "win"}, boxes(21, 5)) == {}


def test_favourites_can_go_in_any_multi(monkeypatch, tmp_path):
    from punt_lab.nba import odds
    fake_world(monkeypatch, tmp_path, spread=-8.5)  # LAL about a 76% chance
    games = espn.slate_games(None)
    h2h = [{"game": "POR @ LAL", "team": "Los Angeles Lakers", "books": {"neds": 1.32, "tab": 1.30, "sportsbet": 1.40}}]
    wins = build.match_winners(games, h2h)
    lal = next(l for l, r in wins if l.team == "LAL")
    assert lal.odds == 1.31 and lal.extra["books"] == {"neds": 1.32, "tab": 1.30}  # Sam's apps only
    assert not lal.extra.get("long_only")
    # below $1.20 it isn't used
    assert not [l for l, _ in build.match_winners(games, [{**h2h[0], "books": {"neds": 1.12}}]) if l.team == "LAL"]
    # in a build it is listed on the legs page
    monkeypatch.setattr(odds, "fetch_h2h", lambda g: h2h)
    monkeypatch.setattr("sys.argv", ["build", "--date", "2026-11-07"])
    build.main()
    legs = json.loads((tmp_path / "legs.json").read_text())
    assert any(l["leg_id"] == "lal_win" and l["best_price"] == 1.32 for l in legs)


def test_win_chance_blends_head_to_head_prices_and_rates_a_value_underdog(monkeypatch, tmp_path):
    from punt_lab.nba import odds as odds_mod
    fake_world(monkeypatch, tmp_path, spread=-8.5)
    games = espn.slate_games(None)
    # every bookie prices both sides: LAL about 80% with the margin removed, against 76% from the spread
    h2h = [{"game": "POR @ LAL", "team": "Los Angeles Lakers", "books": {"neds": 1.20, "tab": 1.20, "sportsbet": 1.20}},
           {"game": "POR @ LAL", "team": "Portland Trail Blazers", "books": {"neds": 4.60, "tab": 4.60, "sportsbet": 4.60}}]
    wins = {l.team: (l, r) for l, r in build.match_winners(games, h2h)}
    market = (1 / 1.20) / (1 / 1.20 + 1 / 4.60)
    assert abs(wins["LAL"][1].model_prob - (0.7 * market + 0.3 * build.norm_cdf(8.5 / 12))) < 1e-9
    assert "POR" not in wins  # about 21%: not rated
    # ESPN and the odds feed name some teams differently
    assert odds_mod.nickname("LA Clippers") == odds_mod.nickname("Los Angeles Clippers") == "clippers"
    # CLE v PHI is close to even; PHI is rated only when Sam's apps pay over the odds
    even = [{"game": "PHI @ CLE", "team": "Cleveland Cavaliers", "books": {"sportsbet": 1.80, "unibet": 1.80, "neds": 1.75}},
            {"game": "PHI @ CLE", "team": "Philadelphia 76ers", "books": {"sportsbet": 2.05, "unibet": 2.05, "neds": 2.05}}]
    assert not {l.team for l, _ in build.match_winners(games, even)} & {"PHI", "CLE"}
    even[1]["books"]["neds"] = 2.40
    phi = next(l for l, _ in build.match_winners(games, even) if l.team == "PHI")
    assert phi.extra["side"] == "underdog" and phi.odds == 2.40


def test_a_less_likely_leg_needs_clearly_better_value_to_be_used():
    from punt_lab.nba import pick
    from punt_lab.nba.model import Leg, LegResult

    def cand(n, prob, price, game):
        leg = Leg(leg_id=f"p{n}", player=f"P{n}", team=f"T{n}", game=game, stat="points", line=10, log=[12.0] * 5,
                  prior_mean=12, prior_games=0, odds=price, extra={"books": {"neds": price}})
        return leg, LegResult(hit_rate=prob, context_shift=0.0, matchup=1.0, minutes_factor=1.0, proj_mean=12.0, proj_sd=3.0,
                              market_mean=None, fair_price=1 / prob, model_prob=prob, implied=None, edge=None)

    base = [cand(n, 0.80 - n * 0.002, 1.20, f"G{n % 8}") for n in range(16)]   # value about 0.95 each
    fair = cand(90, 0.60, 1.60, "G90")    # value 0.96: not clearly better, never searched
    great = cand(91, 0.60, 1.90, "G91")   # value 1.14: clearly better
    used = lambda cands: {cands[i][0].leg_id for k, v in pick.choose(cands).items() if k in ("Safest", "Best value", "3-leg value") for i in v}
    assert "p90" not in used(base + [fair])
    assert "p91" in used(base + [great])



def test_closing_prices_are_saved_and_graded(monkeypatch, tmp_path):
    from punt_lab.nba import late, oddspapi
    fake_world(monkeypatch, tmp_path)
    monkeypatch.setattr("sys.argv", ["build", "--date", "2026-11-07"])
    build.main()
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["recheck_at"] and "rechecked_at" not in report      # for the site's "ready to bet?" box
    assert {"Top pick", "Second pick"} <= {m["multi"] for m in json.loads((tmp_path / "multis.json").read_text())}
    legs = json.loads((tmp_path / "legs.json").read_text())
    leg = next(l for l in legs if l["stat"] != "win")
    for l in legs:                                                    # the morning's Neds price
        if l["leg_id"] == leg["leg_id"]:
            l["open_books"] = {"neds": 1.50}
    (tmp_path / "legs.json").write_text(json.dumps(legs))
    calls = []
    monkeypatch.setattr(oddspapi, "enabled", lambda: True)
    monkeypatch.setattr(oddspapi, "fetch", lambda starts, books=None: calls.append(1) or {
        "market": {}, "books": {(leg["player"].lower(), leg["stat"], int(leg["line"])): {"neds": 1.40, "unibet": 1.6}}})
    start = espn.slate_games(None)[0]["start"]
    assert late.run(build.date(2026, 11, 7), start - timedelta(minutes=45))
    after = next(l for l in json.loads((tmp_path / "legs.json").read_text()) if l["leg_id"] == leg["leg_id"])
    assert after["open_books"] == {"neds": 1.50} and after["close_books"] == {"neds": 1.40}   # only Sam's apps
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["close_fetches"] == 1 and not report.get("late_refreshes") and len(calls) == 1
    monkeypatch.setattr(oddspapi, "enabled", lambda: False)
    monkeypatch.setattr("sys.argv", ["build", "--date", "2026-11-08"])
    build.main()
    row = next(r for r in json.loads((tmp_path / "results.json").read_text()) if r["leg_id"] == leg["leg_id"])
    assert (row["open_price"], row["close_price"], row["close_book"]) == (1.50, 1.40, "neds")

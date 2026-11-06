from datetime import date
from pathlib import Path

from punt_lab.nba import espn, overrides, pick
from punt_lab.nba.model import Leg, evaluate, multi_chance, no_vig_over


def leg(**kw):
    base = dict(leg_id="x", player="P", team="LAL", game="POR @ LAL", stat="points", line=20,
                log=[23, 28, 29, 26, 16, 21, 19, 27, 21], prior_mean=23.3, prior_games=51)
    base.update(kw)
    return Leg(**base)


def test_matches_dashboard_numbers():
    # Values the dashboard showed for 7 Nov 2026
    reaves = leg(context_log=[23, 29, 26, 27, 21], odds=1.57)
    r = evaluate(reaves)
    assert abs(r.model_prob - 0.80) <= 0.02   # recent-form weighting shifts it slightly
    assert 0.12 <= r.edge <= 0.19

    clingan = leg(stat="rebounds", line=10, log=[7, 11, 10, 8, 9, 18, 11], prior_mean=11.6,
                  prior_games=77, context_log=[9, 18, 11], odds=1.42, team="POR")
    assert abs(evaluate(clingan).model_prob - 0.75) <= 0.03

    kessler = leg(stat="rebounds", line=8, log=[9, 8, 7, 8, 9, 13, 15, 15, 8], prior_mean=10.8, prior_games=5)
    assert abs(evaluate(kessler).model_prob - 0.81) <= 0.03


def test_recent_games_count_more():
    hot = evaluate(leg(log=[15, 15, 15, 15, 25, 25, 25, 25, 25])).proj_mean
    cold = evaluate(leg(log=[25, 25, 25, 25, 15, 15, 15, 15, 15])).proj_mean
    assert hot > cold


def test_venue_and_pace():
    base = evaluate(leg()).proj_mean
    assert evaluate(leg(venue_shift=1.0)).proj_mean > base
    assert evaluate(leg(pace_factor=1.03)).proj_mean > base


def test_minutes_and_matchup_move_the_projection():
    base = evaluate(leg()).proj_mean
    assert evaluate(leg(avg_minutes=36, proj_minutes=30)).proj_mean < base
    assert evaluate(leg(opp_factor=1.10)).proj_mean > base
    # head-to-head is shrunk by roster overlap: 0 overlap = no effect
    assert evaluate(leg(h2h_log=[35, 35], roster_overlap=0.0)).proj_mean == base


def test_market_blend_and_fair_price():
    r = evaluate(leg(market_line=22.5, market_p_over=0.5))
    assert r.fair_price and 1.0 < r.fair_price < 2.0
    assert abs(no_vig_over(-110, -110) - 0.5) < 1e-9


def test_correlation():
    a = leg(stat="rebounds", line=8, team="LAL")
    b = leg(stat="rebounds", line=10, team="POR")
    c = leg(game="PHI @ CLE", team="PHI")
    assert multi_chance([a, b], [0.8, 0.75]) < 0.8 * 0.75
    assert abs(multi_chance([a, c], [0.8, 0.75]) - 0.6) < 1e-9


def test_picker_respects_rules():
    cands = []
    specs = [("A", "LAL", "POR @ LAL", "points", 1.57), ("B", "POR", "POR @ LAL", "rebounds", 1.42),
             ("C", "LAL", "POR @ LAL", "rebounds", 1.25), ("D", "CLE", "PHI @ CLE", "points", None),
             ("E", "PHI", "PHI @ CLE", "points", None), ("F", "BOS", "BOS @ CHA", "assists", 1.10)]
    for name, team, game, stat, price in specs:
        l = leg(leg_id=name, player=name, team=team, game=game, stat=stat, odds=price,
                line=8 if stat != "points" else 20,
                log=[9, 10, 8, 11, 9, 12, 10, 9] if stat != "points" else [24, 22, 26, 21, 23, 25, 22, 24])
        cands.append((l, evaluate(l)))
    chosen = pick.choose(cands)
    names = lambda key: {cands[i][0].player for i in chosen[key]}
    for key in ("Safest", "Best value", "3-leg value", "4-leg $2+", "3-leg $2+", "4-leg $3+", "4-leg $4+"):
        assert "F" not in names(key)            # $1.10 is under the $1.20 floor
        assert not {"B", "C"} <= names(key)     # never two rebound legs from one game
    assert len(chosen["3-leg value"]) == 3 and len(chosen["Safest"]) == 4


def test_overrides(tmp_path: Path):
    f = tmp_path / "o.txt"
    f.write_text("Walker Kessler: skip\nLuka Doncic: minutes 30\n"
                 "date: 2026-11-07\nAustin Reaves 20+ points: odds $1.57\n"
                 "date: 2026-11-08\nDeni Avdija 20+ points: odds 1.80\n")
    ov = overrides.load(f, date(2026, 11, 7))
    assert "walker kessler" in ov.skip and ov.minutes["luka doncic"] == 30
    assert ov.odds == {("austin reaves", 20, "points"): 1.57}


def test_espn_parsers(monkeypatch):
    box = {"boxscore": {"players": [{"team": {"abbreviation": "LAL"}, "statistics": [{
        "labels": ["MIN", "FG", "3PT", "FT", "OREB", "DREB", "REB", "AST", "STL", "BLK", "TO", "PF", "+/-", "PTS"],
        "athletes": [
            {"athlete": {"id": "1", "displayName": "Luka Doncic", "position": {"abbreviation": "PG"}},
             "starter": True, "stats": ["36", "14-24", "6-15", "8-10", "0", "7", "7", "13", "2", "0", "2", "2", "+26", "42"]},
            {"athlete": {"id": "2", "displayName": "Bench Guy"}, "didNotPlay": True, "stats": []},
        ]}]}]}}
    monkeypatch.setattr(espn, "get", lambda url, params=None: box)
    rows = espn.box_score("e1")["LAL"]
    assert rows[0]["pts"] == 42 and rows[0]["threes"] == 6 and rows[0]["min"] == 36 and not rows[0]["dnp"]
    assert rows[1]["dnp"]
    assert espn.parse_spread("LAL -3.5") == ("LAL", -3.5)


def test_questionable_players_stay_off_multis():
    cands = []
    for name, game, risky in [("A", "G1", True), ("B", "G2", False), ("C", "G3", False), ("D", "G4", False), ("E", "G5", False)]:
        l = leg(leg_id=name, player=name, game=game, log=[24, 22, 26, 21, 23, 25, 22, 24])
        l.extra["risky"] = risky
        cands.append((l, evaluate(l)))
    chosen = pick.choose(cands)
    for key in ("Safest", "Best value", "3-leg value", "4-leg $2+", "3-leg $2+", "4-leg $3+", "4-leg $4+"):
        assert "A" not in {cands[i][0].player for i in chosen[key]}


def test_surname():
    from punt_lab.nba.build import surname
    assert surname("Trey Murphy III") == "Murphy" and surname("Darius Acuff Jr.") == "Acuff"



def test_long_multi_legs_are_safe_and_priced():
    from punt_lab.nba import pick
    from punt_lab.nba.model import Leg, evaluate
    cands = []
    for k in range(20):
        leg = Leg(leg_id=f"p{k}", player=f"Player {k}", team=f"T{k % 10}", game=f"G{k % 10}", stat="points",
                  line=15, log=[19, 20, 17, 22, 19, 17, 21, 18], prior_mean=19, prior_games=60,
                  market_line=16.5 + 0.5 * (k % 3), market_p_over=0.5, market_source="test", extra={})
        cands.append((leg, evaluate(leg)))
    ids, note = pick._long_multi(cands)
    assert ids and len(ids) >= 5
    for i in ids:
        assert cands[i][1].model_prob >= pick.LONG_MIN_PROB
        assert pick._assumed_price(*cands[i]) >= pick.LONG_MIN_PRICE

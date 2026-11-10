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
        assert "A" not in {cands[i][0].player for i in chosen.get(key, [])}


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


def test_long_shot_and_mid_range():
    from punt_lab.nba import pick
    from punt_lab.nba.model import Leg, evaluate
    cands = []
    for k in range(24):
        l = Leg(leg_id=f"p{k}", player=f"Player {k}", team=f"T{k % 12}", game=f"G{k % 12}", stat="points",
                line=15 + (k % 4), log=[19, 20, 17, 22, 19, 17, 21, 18], prior_mean=19, prior_games=60, extra={})
        l.odds = 1.25 + 0.05 * (k % 4)
        cands.append((l, evaluate(l)))
    chosen = pick.choose(cands)
    shot = chosen["Long shot $50"]
    price = 1.0
    for i in shot:
        price *= pick._assumed_price(*cands[i])
    assert 5 <= len(shot) <= pick.SHOT_MAX_LEGS
    assert price >= pick.SHOT_TARGET or len(shot) == pick.SHOT_MAX_LEGS
    probs = [cands[i][1].model_prob for i in shot]
    assert probs == sorted(probs, reverse=True)           # most likely legs first
    mid = chosen["Mid-range $8+"]
    assert len(mid) in pick.MID_LEGS and len({cands[i][0].player for i in mid}) == len(mid)


def test_every_multi_pays_two_dollars_and_one_per_game():
    from punt_lab.nba import pick
    from punt_lab.nba.model import Leg, evaluate
    cands = []
    for k in range(24):
        l = Leg(leg_id=f"p{k}", player=f"Player {k}", team=f"T{k % 12}", game=f"G{k % 6}", stat="points",
                line=15 + (k % 4), log=[19, 20, 17, 22, 19, 17, 21, 18], prior_mean=19, prior_games=60, extra={})
        l.odds = 1.22 + 0.04 * (k % 4)
        cands.append((l, evaluate(l)))
    chosen = pick.choose(cands)
    for name, ids in chosen.items():
        if name.startswith("_"):
            continue
        price = 1.0
        for i in ids:
            price *= pick._assumed_price(*cands[i])
        assert price >= pick.MIN_MULTI, name
    games = [cands[i][0].game for i in chosen["One per game"]]
    assert len(games) == len(set(games)) == 6


def _fake(name, game, prob, price, step=False):
    from punt_lab.nba.model import Leg, LegResult
    l = Leg(leg_id=f"{name}{'s' if step else ''}", player=name, team=game, game=game, stat="points", line=20 if step else 15,
            log=[18] * 8, prior_mean=18, prior_games=60, odds=price, extra={"step_up": step})
    return l, LegResult(prob, 0, 1, 1, 18, 4, None, None, prob, 1 / price, prob - 1 / price)


def test_long_shot_uses_higher_lines_to_reach_fifty():
    from punt_lab.nba import pick
    from punt_lab.nba.model import multi_chance
    cands = []
    for k in range(20):
        cands.append(_fake(f"P{k}", f"G{k % 8}", 0.83 - 0.002 * k, 1.20))           # likely, short-priced
        cands.append(_fake(f"P{k}", f"G{k % 8}", 0.62 - 0.004 * k, 1.70, step=True))  # next line up
    chosen = pick.choose(cands)
    shot = chosen["Long shot $50"]
    price = pick._price(cands, shot)
    assert price >= 50 and len(shot) <= pick.SHOT_MAX_LEGS
    assert len({cands[i][0].player for i in shot}) == len(shot)
    # beats simply stacking the most likely legs (which can't even reach $50 in 15 legs here)
    assert pick._price(cands, sorted(range(len(cands)), key=lambda i: -cands[i][1].model_prob)[:15]) < 50
    mid = chosen["Mid-range $8+"]
    assert pick._price(cands, mid) >= 8


def test_mid_range_is_close_to_the_best_possible():
    from itertools import combinations
    from punt_lab.nba import pick
    from punt_lab.nba.model import multi_chance
    cands = []
    for k in range(14):
        cands.append(_fake(f"P{k}", f"G{k % 7}", 0.84 - 0.003 * k, 1.22))           # safe legs
        cands.append(_fake(f"P{k}", f"G{k % 7}", 0.45 - 0.003 * k, 2.10, step=True))  # long next lines
    mid = pick.choose(cands)["Mid-range $8+"]
    got = multi_chance([cands[i][0] for i in mid], [cands[i][1].model_prob for i in mid])
    best = 0.0
    for combo in combinations(range(len(cands)), 6):
        if pick._ok_combo([cands[i][0] for i in combo]) and pick._price(cands, combo) >= 8:
            best = max(best, multi_chance([cands[i][0] for i in combo], [cands[i][1].model_prob for i in combo]))
    assert got >= 0.9 * best


def test_multis_spread_across_games_and_avoid_conflicting_legs():
    from collections import Counter
    from punt_lab.nba.model import LegResult, pair_link
    # one game with lots of attractive legs: the short multis may take two of them at most
    cands = [_fake(f"S{k}", "SA @ PHX", 0.84 - 0.01 * k, 1.30) for k in range(6)]
    cands += [_fake(f"O{k}", f"G{k}", 0.80, 1.25) for k in range(6)]
    chosen = pick.choose(cands)
    for name, ids in chosen.items():
        if name.startswith("_"):
            continue
        worst = max(Counter(cands[i][0].game for i in ids).values())
        assert worst <= (pick.GAME_MAX_SHORT if len(ids) <= pick.SHORT_LEGS else pick.GAME_MAX_LONG), name

    # a team to win never sits with a leg from the team it's playing
    win = Leg(leg_id="sa_win", player="San Antonio Spurs", team="SA", game="SA @ PHX", stat="win", line=1,
              log=[], prior_mean=0, prior_games=0, market="San Antonio Spurs to win")
    suns = leg(leg_id="b", player="Brooks", team="PHX", game="SA @ PHX")
    spur = leg(leg_id="f", player="Fox", team="SA", game="SA @ PHX")
    away = leg(leg_id="n", player="Nembhard", team="IND", game="IND @ DAL")
    assert not pick._ok_combo([win, suns])
    assert pick._ok_combo([win, spur]) and pick._ok_combo([win, away])
    assert pair_link(win, suns) < 0 < pair_link(win, spur)
    assert multi_chance([win, suns], [0.76, 0.6]) < 0.76 * 0.6


def test_underdog_share_lowers_the_prediction():
    base, dog = evaluate(leg()), evaluate(leg(side_factor=0.96))
    assert dog.proj_mean < base.proj_mean and dog.model_prob < base.model_prob


def test_one_game_slate_still_gets_multis():
    cands = [_fake(f"S{k}", "SA @ PHX", 0.80, 1.30) for k in range(8)]
    chosen = pick.choose(cands)
    assert len(chosen["4-leg $2+"]) == 4
    pick.choose([_fake(f"S{k}", f"G{k}", 0.80, 1.30) for k in range(8)])  # back to the normal limits
    assert pick._caps == (pick.GAME_MAX_SHORT, pick.GAME_MAX_LONG)


def test_two_leg_multi_and_verdicts():
    from punt_lab.nba import build
    cands = [_fake(f"P{k}", f"G{k}", 0.62, 1.55) for k in range(4)] + [_fake(f"Q{k}", f"H{k}", 0.84, 1.22) for k in range(6)]
    chosen = pick.choose(cands)
    two = chosen["2-leg $2+"]
    assert len(two) == 2 and pick.combo_price(cands, two) >= 2.0
    # value needs the app price to beat break-even by 5%; 3% short is a skip; no app price is "check"
    base = {"est_hit": 0.5, "break_even": 2.0, "multi": "2-leg $2+", "best_book": "neds", "legs_text": "A; B", "est_price": 2.1}
    rate = lambda price: build.verdict({**base, "best_book_price": price})["verdict"]
    assert (rate(2.20), rate(2.02), rate(1.90), rate(None)) == ("value", "marginal", "skip", "check")
    good = {**base, "best_book_price": 2.2, **build.verdict({**base, "best_book_price": 2.2})}
    bad = {**base, "multi": "4-leg $2+", "best_book_price": 1.8, **build.verdict({**base, "best_book_price": 1.8})}
    assert build.headline_for([bad, good]).startswith("Value today: 2-leg $2+ on Neds $2.20 (needs $2.00)")
    assert build.headline_for([bad]).startswith("Skip today")
    twin = {**good, "multi": "Best value", "leg_ids": ["a", "b"]}
    assert build.headline_for([{**good, "leg_ids": ["b", "a"]}, twin]).count("$2.20") == 1    # same legs, said once
    unpriced = {**base, "best_book_price": None, **build.verdict({**base, "best_book_price": None})}
    assert "No app prices every leg yet" in build.headline_for([unpriced])

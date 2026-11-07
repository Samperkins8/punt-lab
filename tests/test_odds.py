from punt_lab.nba import odds
from punt_lab.nba.build import _multi_books
from punt_lab.nba.model import Leg


def test_parse_au_main_and_ladder():
    data = {"bookmakers": [
        {"key": "sportsbet", "markets": [
            {"key": "player_points", "outcomes": [
                {"description": "Jamal Murray", "name": "Over", "point": 22.5, "price": 1.92},
                {"description": "Jamal Murray", "name": "Under", "point": 22.5, "price": 1.84}]},
            {"key": "player_points_alternate", "outcomes": [
                {"description": "Jamal Murray", "name": "Over", "point": 19.5, "price": 1.46}]}]},
        {"key": "tab", "markets": [
            {"key": "player_points_alternate", "outcomes": [
                {"description": "Jamal Murray", "name": "Over", "point": 19.5, "price": 1.52}]}]}]}
    out = odds.empty()
    odds._parse(data, out)
    assert out["books"][("jamal murray", "points", 20)] == {"sportsbet": 1.46, "tab": 1.52}
    assert out["books"][("jamal murray", "points", 23)] == {"sportsbet": 1.92}  # main line over 22.5 = 23+
    line, p_over, _ = odds.finalise(out)["market"][("jamal murray", "points")]
    assert line == 22.5 and 0.48 < p_over < 0.50          # margin removed
    assert odds.best(out["books"][("jamal murray", "points", 20)]) == ("tab", 1.52)


def test_plan_spends_budget_on_busiest_games_first():
    wanted = {("A", "B"): {"points": 4, "rebounds": 1}, ("C", "D"): {"points": 2}, ("E", "F"): {"assists": 1}}
    todo = odds.plan(wanted, budget=6)
    assert todo == {("A", "B"): ["points"], ("C", "D"): ["points"], ("A", "B"): ["points", "rebounds"]} or \
        todo == {("A", "B"): ["points", "rebounds"], ("C", "D"): ["points"]}


def test_multi_priced_at_one_bookie():
    def leg(books):
        return Leg(leg_id="x", player="x", team="t", game="g", stat="points", line=20, log=[20] * 5,
                   prior_mean=20, prior_games=50, extra={"books": books})
    m = _multi_books([leg({"tab": 1.5, "neds": 1.4}), leg({"tab": 1.3, "neds": 1.45, "sportsbet": 1.6})])
    assert m["best_book"] == "neds" and m["best_book_price"] == round(1.4 * 1.45, 2)  # tab 1.95 < neds 2.03
    assert _multi_books([leg({"tab": 1.5}), leg({})])["best_book"] is None



def test_oddspapi_parse_and_merge():
    from punt_lab.nba import oddspapi
    defs = {111: {"stat": "points", "handicap": 19.5, "outcomes": {"1": "over", "2": "under"}},
            112: {"stat": "points", "handicap": 24.5, "outcomes": {"3": "over", "4": "under"}}}
    markets = {"111": {"outcomes": {"1": {"players": {"9": {"playerName": "Murray, Jamal", "price": 1.50, "mainLine": False}}},
                                    "2": {"players": {"9": {"playerName": "Murray, Jamal", "price": 2.55}}}}},
               "112": {"outcomes": {"3": {"players": {"9": {"playerName": "Murray, Jamal", "price": 2.40}}},
                                    "4": {"players": {"9": {"playerName": "Murray, Jamal", "price": 1.55}}}}}}
    op = {"market": {}, "books": {}}
    assert oddspapi._parse(markets, defs, "neds", op) == 2
    assert op["books"][("jamal murray", "points", 20)] == {"neds": 1.5}
    assert op["market"][("jamal murray", "points")][0][0] == 24.5  # line closest to even money
    other = odds.empty()
    other["books"][("jamal murray", "points", 20)] = {"sportsbet": 1.46}
    merged = odds.finalise(odds.merge(op, other))
    assert merged["books"][("jamal murray", "points", 20)] == {"neds": 1.5, "sportsbet": 1.46}
    assert merged["market"][("jamal murray", "points")][0] == 24.5
    assert oddspapi.player_key("Jackson Jr., Jaren") == "jaren jackson jr."

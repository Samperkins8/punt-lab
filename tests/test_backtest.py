import json
import random

from punt_lab.nba import backtest, espn, model


def fake_espn(monkeypatch, sd_scale):
    """Players whose points really vary `sd_scale` times more than the model's default spread assumes."""
    def roster(tid):
        return [{"id": f"{tid}-{k}", "name": f"P{tid}-{k}", "pos": "G"} for k in range(2)] if int(tid) <= 10 else []

    def gamelog(pid, year):
        rng = random.Random(f"{pid}-{year}")
        mu = 18 + rng.random() * 10
        games = []
        for k in range(45):
            pts = max(0, round(rng.gauss(mu, mu * model.SPREAD["points"] * sd_scale)))
            games.append({"opp": f"T{k % 10}", "team": "X", "pts": pts, "reb": 3, "ast": 2, "threes": 1, "min": 32,
                          "date": f"{year - 1}-11-{1 + k % 28:02d}T0{k // 28}:00Z", "home": k % 2 == 0})
        return {"games": games, "gp": len(games)}

    monkeypatch.setattr(espn, "team_roster", roster)
    monkeypatch.setattr(espn, "last_season_log", gamelog)


def test_backtest_finds_wider_spread_and_applies_it(monkeypatch, tmp_path):
    fake_espn(monkeypatch, sd_scale=1.5)
    saved = backtest.current_settings()
    monkeypatch.setattr(backtest, "CACHE", tmp_path / "cache")
    monkeypatch.setattr(backtest, "OUT", tmp_path / "backtest.json")
    monkeypatch.setattr(backtest, "DATA", tmp_path)
    monkeypatch.setattr(backtest, "MIN_SAMPLES", 200)
    monkeypatch.setattr(model, "TUNED_FILE", tmp_path / "tuned.json")
    monkeypatch.setattr(backtest, "VERSIONS_FILE", tmp_path / "versions.json")
    try:
        backtest.main()
        out = json.loads((tmp_path / "backtest.json").read_text())
        tuned = json.loads((tmp_path / "tuned.json").read_text())
        assert out["applied"]
        assert out["after"]["test"]["log_loss"] < out["before"]["test"]["log_loss"]
        assert tuned["settings"]["SPREAD"]["points"] > saved["SPREAD"]["points"]
        assert list((tmp_path / "cache").glob("gamelogs_*.json.gz"))  # finished seasons cached

        # next week: starts from the tuned settings and must never fall back to the hand-set ones
        model.load_tuned(tmp_path / "tuned.json")
        backtest.main()
        again = json.loads((tmp_path / "tuned.json").read_text())
        assert again["settings"] and again["settings"]["SPREAD"]["points"] > saved["SPREAD"]["points"]
        # every version is kept and re-scored: hand-set first, the one in use last, matching "after"
        out = json.loads((tmp_path / "backtest.json").read_text())
        versions = json.loads((tmp_path / "versions.json").read_text())
        assert len(out["versions"]) == len(versions) + 1 >= 2
        assert out["versions"][0]["log_loss"] == out["before"]["test"]["log_loss"]
        assert abs(out["versions"][-1]["log_loss"] - out["after"]["test"]["log_loss"]) < 1e-9
    finally:
        backtest.apply_settings(saved)
        model.CALIBRATE = True


def test_live_calibration_pulls_overconfident_legs_down():
    rows = [{"market": "20+ points", "model_prob": 0.85, "landed": i % 10 < 7} for i in range(400)]
    rows += [{"market": "20+ points", "model_prob": 0.62, "landed": i % 10 < 6} for i in range(200)]
    curves, summary = backtest.live_calibration(rows)
    pts = dict((round(x, 2), y) for x, y in curves["points"])
    assert 0.70 < pts[0.85] < 0.85          # 85% legs that land 70% get pulled down (shrunk, not all the way)
    assert summary["points"]["legs"] == 600
    model.CALIBRATION = {"points": curves["points"]}
    try:
        assert model.calibrate("points", 0.85) == pts[0.85]
        assert model.calibrate("rebounds", 0.85) == 0.85
    finally:
        model.CALIBRATION = {}


def test_measure_links_finds_rebounders_competing():
    base = {"stat": "rebounds", "vals": [8, 9, 7, 8, 10, 6, 8, 9], "prior_mean": 8, "prior_games": 60,
            "venue_shift": 0.0, "opp_factor": None, "lines": [8], "team": "AAA", "opp": "BBB"}
    data = []
    for k in range(400):
        a_big = k % 2 == 0
        data.append({**base, "pid": "a", "date": f"d{k}", "actual": 11 if a_big else 5})
        data.append({**base, "pid": "b", "date": f"d{k}", "team": "BBB", "opp": "AAA", "actual": 5 if a_big else 11})
    links = backtest.measure_links(data)
    assert links["SAME_GAME_REBOUNDS"]["pairs"] == 400
    assert links["SAME_GAME_REBOUNDS"]["r"] < -0.2


def test_line_settings_are_fitted_from_past_results():
    """Players score less against stronger opponents than usual; the fit should find that and keep it."""
    saved = backtest.current_settings()
    model.CALIBRATE = False
    try:
        def make(seed, n):
            rng = random.Random(seed)
            out = []
            for _ in range(n):
                mu = 16 + rng.random() * 10
                shift = rng.gauss(0, 6)
                true = mu * (1 + 2.0 * shift / model.LEAGUE_TOTAL)
                vals = [max(0, round(rng.gauss(mu, mu * 0.3))) for _ in range(12)]
                out.append({"season": 1, "stat": "points", "vals": vals, "actual": max(0, round(rng.gauss(true, mu * 0.3))),
                            "prior_mean": mu, "prior_games": 60, "venue_shift": 0.0, "opp_factor": None, "pid": "p",
                            "date": "2026-01-01", "team": "A", "opp": "B", "b2b": False, "avg_min": 32,
                            "expected": shift, "shift": shift})
            backtest.choose_lines(out)
            return out
        train, test = make(1, 5000), make(2, 5000)
        assert backtest.has_lines(train)
        settings, report = backtest.tune_lines(train, test, backtest.current_settings())
        assert report["chosen"] != "none" and report["log_loss"][report["chosen"]] < report["log_loss"]["none"]
        assert report["fitted"]["SIDE_WEIGHT"]["points"] >= 1.0
        assert backtest.current_settings() == saved          # tune_lines leaves the settings as it found them
        assert not backtest.has_lines([{**s, "shift": None} for s in train])
    finally:
        backtest.apply_settings(saved)
        model.CALIBRATE = True


def test_team_ratings_stand_in_for_the_line():
    games = lambda team, opp, margin: [{"team": team, "opp": opp, "date": f"2026-01-{d:02d}T00:00Z", "margin": margin, "min": 30} for d in range(1, 11)]
    logs = {"a": {"games": games("GOOD", "BAD", 12)}, "b": {"games": games("BAD", "GOOD", -12)}}
    r = backtest.TeamRatings(logs)
    expected, shift = r.line("BAD", "GOOD", "2026-02-01", home=False)
    assert expected < -10 and shift < -5          # big underdogs, against a much better side than average
    assert r.line("BAD", "GOOD", "2026-01-02", home=True) == (None, None)   # too early in the season to say
    assert model.side_factor("points", shift) < 1 == model.side_factor("rebounds", shift)
    assert model.blowout_cut(expected) == model.UNDERDOG_CUT_BIG and model.blowout_cut(3) == 1.0
    assert model.blowout_cut(12) == model.BLOWOUT_CUT and model.blowout_cut(-8) == model.UNDERDOG_CUT
    assert espn._margin({"homeTeamScore": "118", "awayTeamScore": "104"}, home=False) == -14
    assert espn._margin({"score": "W 121-110", "gameResult": "W"}, home=None) == 11 and espn._margin({}, True) is None


def test_record_version_skips_link_only_changes():
    d = {"SPREAD": {"points": 0.3}, "SAME_GAME_OTHER": 0.05}
    v = backtest.record_version([], "2026-11-01", {"SPREAD": {"points": 0.4}, "SAME_GAME_OTHER": 0.05}, {}, d)
    v = backtest.record_version(v, "2026-11-08", {"SPREAD": {"points": 0.4}, "SAME_GAME_OTHER": 0.09}, {}, d)
    assert len(v) == 1
    v = backtest.record_version(v, "2026-11-15", {"SPREAD": {"points": 0.4}}, {"points": [[0.5, 0.5]]}, d)
    assert [x["curve"] for x in v] == ["none", "stat"]

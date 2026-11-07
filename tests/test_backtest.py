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

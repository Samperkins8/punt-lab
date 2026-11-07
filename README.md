# Punt Lab

Daily player-prop multi suggestions, built from free data and published as a website.
One folder per sport; NBA is the first.

**Website:** GitHub Pages, from the `docs/` folder.

## NBA

Runs on GitHub Actions (`.github/workflows/nba.yml`):

- **7:44am Melbourne** (summer time): grades yesterday's legs and multis, analyses every game on today's slate, picks three multis.
- **9:38am Melbourne**: re-checks injury news before the first tip and swaps out legs if needed.
- **Every 30 minutes, ~8am–3:30pm**: late check of each game 15–75 minutes before its tip. Marks late outs and warns on affected multis (they're not rebuilt, in case you've bet them), re-analyses the game, and refreshes prices only when someone's status changed (at most twice a day).

You can also run it by hand: Actions → NBA daily multi → Run workflow.

### Your overrides

Edit `overrides/nba.txt` (works in the GitHub app or browser):

```
Walker Kessler: skip
Luka Doncic: minutes 30
Ja Morant: out

date: 2026-11-07
Austin Reaves 20+ points: odds 1.57
```

Lines under a `date:` header apply only to that day. The next run picks them up.

### Optional keys (Settings → Secrets and variables → Actions)

| Secret | What it adds | Cost |
|---|---|---|
| `ODDS_API_KEY` | TAB, Neds and Bet365 prices for each leg (only those apps are kept; change with a `PUNT_BOOKS` variable) and their over/under lines (fair prices), from the-odds-api.com. Only the games and stats today's legs need are priced, within `ODDS_DAILY_CREDITS` (16 a day fits the free 500 a month) | Free plan, 500 credits a month |
| `ODDSPAPI_KEY` | Every game's player prices at Neds, from oddspapi.io: one request for the whole slate (~30 of the free 250 a month) | Free plan |
| `ANTHROPIC_API_KEY` | Claude reads vague injury notes (minutes limits, coming off the bench) | Pay-as-you-go, a few cents a month |

Without them everything still runs on free ESPN data and rules.

### Model

See "How the multis are picked" at the bottom of the NBA page, or `punt_lab/nba/model.py`.
All settings are named constants at the top of that file; tune them once the track record has a few weeks of results.

### Weekly tune

Every Monday at about 12:15pm Melbourne (`.github/workflows/tune.yml`, or run it by hand under Actions → NBA weekly tune):

- **Backtest**: replays the last two seasons and this one so far through the model, using only what was known before each game, and scores every leg against what happened.
- **Tune**: adjusts the model settings on the older season and keeps them only if they also predict better on the newer seasons. Saved to `punt_lab/nba/tuned.json`, which the daily run loads.
- **Live correction**: once a stat has 150+ graded daily legs, a calibration curve nudges its percentages towards what actually landed.

Results are on the NBA page under "How honest are the percentages?". Finished seasons' ESPN game logs are cached in `punt_lab/nba/cache` so later runs are quick.

### Tests

```
pip install -r requirements.txt pytest
python -m pytest -q tests
```

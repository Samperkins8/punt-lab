# Punt Lab

Daily player-prop multi suggestions, built from free data and published as a website.
One folder per sport; NBA is the first.

**Website:** GitHub Pages, from the `docs/` folder.

## NBA

Runs on GitHub Actions (`.github/workflows/nba.yml`):

- **7:44am Melbourne** (summer time): grades yesterday's legs and multis, analyses every game on today's slate, picks three multis.
- **9:38am Melbourne**: re-checks injury news before the first tip and swaps out legs if needed.

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
| `ODDS_API_KEY` | US market lines (fair prices) and Sportsbet prices where available, from the-odds-api.com | Free plan, limited monthly credits |
| `ANTHROPIC_API_KEY` | Claude reads vague injury notes (minutes limits, coming off the bench) | Pay-as-you-go, a few cents a month |

Without them everything still runs on free ESPN data and rules.

### Model

See "How the multis are picked" at the bottom of the NBA page, or `punt_lab/nba/model.py`.
All settings are named constants at the top of that file; tune them once the track record has a few weeks of results.

### Tests

```
pip install -r requirements.txt pytest
python -m pytest -q tests
```

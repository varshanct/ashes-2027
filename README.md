# Forecasting the 2027 Ashes

A Bayesian forecast of the 2027 men's Ashes: England v Australia, five Tests in England from 18 June to 2 August 2027.
For each Test the model gives the chance of an England win, a draw and an Australia win, and carries its uncertainty
through to every series result.

## Headline forecast

![Model forecast and my own forecast, Test by Test](figures/forecast_2027.png)

| Test | Ground | Dates | England win | Draw | Australia win |
| --- | --- | --- | --- | --- | --- |
| 1 | Trent Bridge | 18–22 June | 46% | 10% | 44% |
| 2 | Lord's | 30 June–4 July | 43% | 15% | 41% |
| 3 | Edgbaston | 8–12 July | 46% | 11% | 44% |
| 4 | Utilita Bowl | 21–25 July | 43% | 15% | 41% |
| 5 | Kia Oval | 29 July–2 August | 46% | 10% | 44% |

| Series | England win | Drawn series | Australia win | Australia retain the urn |
| --- | --- | --- | --- | --- |
| Model | 45% | 13% | 42% | 55% |
| Model, if the Tests were independent | 45% | 14% | 41% | 55% |
| My own forecast, written before any modelling | 42% | 16% | 42% | 58% |

England are slight favourites at home because the model's home advantage (0.62) is a little larger than the gap
between the teams (Australia 1.35, England 0.77 on the model's scale).

## The question and the approach

1. **Read every men's Test** in Cricsheet since 2001, ball by ball, into tested tables of matches, innings, deliveries
   and appearances.
2. **Check the data independently.** Every recorded winning margin must match the totals rebuilt from individual
   deliveries. It holds for all 729 wins with a recorded margin.
3. **Profile the five grounds** from their history: draw rates, first-innings scores, overs bowled, the toss and how
   batters were dismissed.
4. **Fit a Bayesian model** in PyMC with Hamiltonian Monte Carlo (NUTS), learning from both the result and the
   winning margin of every Test.
5. **Back-test it** on the 2019 and 2023 Ashes, using only Tests played before each series, with and without margins.
6. **Forecast each 2027 Test and the series**, and compare with a forecast written down before any modelling.

## The model

Each Test is an ordered outcome: team A loses, draws or wins. The strength gap plus home advantage sits on one line,
and a band of width $w$ in the middle is a draw.

$$\eta = s_{A,t} - s_{B,t} + h \cdot \text{home}$$

$$P(\text{A loses}) = \sigma\left(-\tfrac{w}{2} - \eta\right), \qquad P(\text{A wins}) = 1 - \sigma\left(\tfrac{w}{2} - \eta\right)$$

$$\log w = \beta_0 + v_{\text{venue}} + \tau \cdot \frac{\text{year} - 2019}{10}$$

For every win, the size of the win $m$ (in wickets) also depends on the winner's edge $\eta_{\text{winner}}$, through
a normal distribution cut off at 0:

$$\log(1 + m) \sim \text{Normal}^{+}\left(\mu_0 + \lambda \cdot \eta_{\text{winner}},\ \sigma_m\right)$$

| Part | Choice |
| --- | --- |
| Team strength $s_{T,t}$ | A yearly random walk for each of 11 teams, centred each year so 0 is an average Test side |
| Home advantage $h$ | One value for all teams. The home side is inferred from the data: the team playing at least 80% of the Tests at a place |
| Draw band | A base width, a trend per decade since 2019, and a venue effect for each of 103 venues, partially pooled |
| Winning margin $m$ | Wickets in hand, or runs divided by the average runs per wicket (32.6). An innings win adds 10 wickets. |
| Parameterisation | Non-centred, for the random walk and the venue effects |
| Data | 898 Tests from 2001 to 2026, 728 of them wins with a recorded margin (the one-off ICC World XI Test is excluded) |

## Does the fit work?

| Check | Result | Target |
| --- | --- | --- |
| Worst R-hat | 1.006 | below 1.01 |
| Smallest effective sample size | 1,001 | above 400 |
| Divergences | 0 | 0 |

Four chains of 1,000 warm-up and 1,000 kept draws, with a target acceptance of 0.95. R-hat and effective sample size are
implemented from first principles in [`diagnostics.py`](src/cricket/diagnostics.py) and tested against cases with known
answers. Full results are in [`figures/sampler_checks.csv`](figures/sampler_checks.csv).

## How good is it?

I tested two versions of the model on the 2019 and 2023 Ashes, fitting each only on Tests played before the series
began: one that sees only results, and one that also sees winning margins.

| Average per Test (lower is better) | With margins | Results only | Past results | Equal thirds |
| --- | --- | --- | --- | --- |
| Log loss | 1.048 | 1.061 | 1.097 | 1.099 |
| Brier score | 0.637 | 0.647 | 0.670 | 0.667 |

The version with margins did best in both series. Before the 2019 and 2023 Ashes it gave England about an even chance of
winning the series (48% and 50%), where the results-only version gave 60% and 54%; both series finished 2-2. Ten
Tests is still a small sample. Details are in [`figures/backtest_tests.csv`](figures/backtest_tests.csv) and
[`DECISIONS.md`](DECISIONS.md).

![Back-test: model chances before each Ashes Test, and what happened](figures/backtest.png)

## What the model learned

| Parameter | Mean | 90% interval |
| --- | --- | --- |
| Home advantage | 0.62 | 0.51 to 0.73 |
| Draw trend per decade | −0.53 | −0.71 to −0.35 |
| Margin slope $\lambda$ | 0.17 | 0.14 to 0.21 |
| Yearly step in team strength | 0.15 | 0.08 to 0.23 |
| Australia strength, 2026 | 1.35 | 0.89 to 1.81 |
| England strength, 2026 | 0.77 | 0.34 to 1.21 |

- **Stronger winners win by more.** The margin slope is clearly above 0, so every win now says how big the gap
  between the teams was, not only who won.
- **Team strength now moves.** With margins, the model allows almost twice as much change each year. England's peak
  around 2010–11 and Australia's dip around 2010–13 both show, where the results-only model was almost flat.
- **Draws are becoming much rarer.** The draw band narrows by about 40% per decade. At the five grounds, 20% of Tests
  were drawn overall but 13% since 2019.
- **Grounds differ beyond their weather.** Lord's draws more than any other well-sampled ground despite London being
  one of the drier fixtures; Trent Bridge, the wettest, draws among the least.
- **Partial pooling at work.** The Utilita Bowl has a raw draw rate of 43% from only seven Tests. The model rates it
  about the same as Lord's, with a 90% interval about 50% wider.

![Team strength over time](figures/team_strength.png)

## How each ground plays

![How the five grounds have played](figures/ground_profiles.png)

![How the five grounds have played since 2019](figures/ground_profiles_since_2019.png)

![England v Australia at each ground](figures/ashes_at_grounds.png)

## Limitations

- **Still smoothed.** Strengths change from year to year now, but with about ten Tests per team per year they are still
  smoothed. Australia's 4-1 win in 2025-26 barely shows.
- **A simple margin rule.** Converting runs to wickets at one average rate, and adding 10 wickets for an innings win,
  is a judgement, not a measured fact.
- **Teams, not players.** Squads, injuries and workloads between back-to-back Tests are collected and validated in
  `data/manual/` but not yet used by the model.
- **One home advantage for every team.**
- **No historical match weather.** Each ground's effect already includes its typical summer weather. The Met Office
  climate averages in `data/manual/` explain those effects and can adjust a forecast in the week of each Test.
- **A small back-test.** Ten Tests is too few to rank models with confidence.

## Next steps

1. Use ball-by-ball runs and wickets, so the model can follow changes in form within a season.
2. Back-test on every year of Tests, not only two Ashes series, to give hundreds of scored forecasts.
3. Build player-level strengths from the expected XIs, with availability per Test and a cover plan for replacements.

## Reproduce it

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run pytest                                      # 168 tests
```

Download the **Test matches (JSON)** and the **People register** from [Cricsheet](https://cricsheet.org/downloads/).
Unzip the matches into `data/raw/tests_json/` and save `people.csv` in `data/raw/`. Then:

```bash
uv run python -m cricket.ingest                    # build the tables
uv run python -m cricket.checks                    # independent check on the totals
uv run python -m cricket.manual_data               # validate squads, availability, fixtures and weather
uv run python -m cricket.cover_plan                # validate the replacement plan
uv run python -m cricket.grounds                   # ground profiles and charts
uv run python -m cricket.grounds --since 2019-01-01
uv run python -m cricket.backtest --tune 2000 --draws 2000   # back-test both versions
uv run python -m cricket.forecast                  # fit the model and forecast 2027
uv run python -m cricket.forecast --results-only   # the same without winning margins
```

## Project layout

| Path | Contents |
| --- | --- |
| `src/cricket/ingest.py` | Reads Cricsheet Test files into four tables |
| `src/cricket/checks.py` | Independent check of innings totals against recorded margins |
| `src/cricket/manual_data.py`, `cover_plan.py` | Validation of squads, availability, fixtures, venues, weather and the cover plan |
| `src/cricket/find_ids.py`, `fill_ids.py` | Helpers for matching players to Cricsheet IDs |
| `src/cricket/grounds.py`, `charts.py` | Ground profiles and charts |
| `src/cricket/model_data.py` | Prepares the model inputs, including the home side and winning margins |
| `src/cricket/bayes_model.py` | The PyMC model |
| `src/cricket/diagnostics.py` | R-hat and effective sample size |
| `src/cricket/predict.py` | Test and series forecasts from posterior draws |
| `src/cricket/backtest.py` | Forecasts the 2019 and 2023 Ashes using only earlier Tests, and scores both versions |
| `src/cricket/forecast.py` | Fits the model and produces the forecast |
| `data/manual/` | Squads, availability, cover plan, fixtures, venues, climate and my own forecast |
| `figures/` | Charts and results tables |
| `tests/` | 168 tests |
| [`DECISIONS.md`](DECISIONS.md) | Every modelling and data decision, with reasons |

## Data sources

- Ball-by-ball match data: [Cricsheet](https://cricsheet.org)
- 2027 schedule: [ECB](https://www.ecb.co.uk/news/4539854/three-ashes-series-headline-england-cricket-home-international-fixtures-for-2027)
- Climate averages, 1991–2020: [Met Office](https://www.metoffice.gov.uk/research/climate/maps-and-data/location-specific-long-term-averages)
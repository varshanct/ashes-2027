# Project conventions

Project: forecast the 2027 men's Ashes (England v Australia, five Tests, June to August 2027).

- Never edit anything in `data/raw/`. All changes to data happen in code.
- Every model input must use only information available before the match it describes. No future data.
- Train/test splits are by date, never random.
- Use Cricsheet player IDs, never player names, to join tables.
- Team names go through `cricket.ingest.clean_team`.
- A Test has three outcomes: home win, draw, away win. Never drop draws.
- Every new function gets a pytest test that would fail if the function were wrong.
- Run `uv run pytest` before every commit.
- Record any modelling or data decision in `DECISIONS.md`.
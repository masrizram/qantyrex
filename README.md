# trading_bot

Production-grade autonomous quantitative trading research, validation, and
execution platform. See the project prompt for the full design specification.

## Status

This is a research / paper-trading platform. It does **not** guarantee profit
and will **fail closed** if live-trading acceptance gates are not met.

## Quickstart

```bash
python -m venv .venv
. .venv/Scripts/activate   # Windows
pip install -r requirements.txt
cp .env.example .env       # then edit
pytest -q
```

## Safety

- `LIVE_TRADING_ENABLED=false` by default.
- A live order is only submitted when `TRADING_MODE=LIVE` **and**
  `LIVE_TRADING_ENABLED=true` **and** all acceptance gates pass.
- No strategy is "promoted" without positive out-of-sample expectancy and
  passing Monte Carlo / walk-forward / sensitivity / stress gates.

## Layout

See the project prompt for the full module map. Major subsystems:
`core/`, `data/`, `strategy/`, `risk/`, `execution/`, `backtest/`,
`research/`, `monitoring/`, `telegram/`, `storage/`.

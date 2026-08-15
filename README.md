# KLAS Kalshi High Model — v0.14

KLAS-only research model for the official Las Vegas daily high used by Kalshi weather markets.

The project is deliberately **simple on screen and detailed underneath**. The validated temperature model stays anchored to the NWS morning high and KLAS intraday behavior, while live weather intelligence watches for forecast/radar/convective conditions that can make the normal heating curve unreliable.

## What v0.13 adds

### Forward-looking NWS hourly forecast
The live update now checks the official NWS point forecast for KLAS and summarizes the rest of today's:

- precipitation probability
- thunderstorm wording
- hourly temperature path
- wind
- forecast sky cover from NWS grid data

### NWS Las Vegas Area Forecast Discussion
The latest VEF Area Forecast Discussion is checked for useful convective language such as:

- monsoon / monsoonal moisture
- thunderstorms / convection
- outflow / gust fronts
- microbursts / DCAPE
- virga
- cloud debris

The dashboard shows only a short signal and excerpt rather than the full discussion.

### MRMS radar around KLAS
The live update checks the official NWS time-enabled MRMS base-reflectivity image service.

A coarse ring scan samples radar pixels around KLAS at approximately 10, 25, and 50 miles and compares the latest scan with one about 30 minutes earlier. It can flag nearby/increasing radar echoes. The dashboard also displays the live MRMS radar image centered on KLAS.

**Important:** this is intentionally a coarse proximity/trend check. It is not yet a full storm-cell tracker, dBZ classifier, or motion-vector algorithm.

### Weather intelligence is separated from the validated temperature model
This matters. The NWS hourly forecast, AFD and radar currently affect:

- weather-risk level
- confidence
- WAIT / WATCH status

They **do not directly alter the predicted high yet**. We will only let new weather variables change the high after they are backtested historically. This prevents us from damaging the 2022–2026 model that already beat the NWS baseline out of sample at later checkpoints.

### Simpler live screen
The page now includes:

- Current KLAS precise temperature
- NWS morning high
- Our model high
- 80% model range
- 6-hour ASOS maximum
- Precise public METAR peak
- overall research status
- four small risk indicators: observed / forecast / AFD / radar
- forecast rain/thunder summary
- live radar panel
- short "why model is here" explanation
- today's hourly progression
- Kalshi bucket probabilities and bid/ask comparison

## Hourly update timing

GitHub Actions remains scheduled for `:05` after every hour. That lets the routine hourly METAR arrive before the dashboard refreshes. If an ASOS six-hour maximum group is present in the METAR, it is parsed automatically and used as a floor for the final-high probability distribution.

## Data sources used live

1. KLAS ASOS/METAR/SPECI observations via the IEM ASOS archive endpoint
2. NWS pre-06:00 Las Vegas PFM high forecast
3. NWS API hourly/grid forecast for KLAS
4. NWS Las Vegas (VEF) Area Forecast Discussion
5. NWS MRMS time-enabled base-reflectivity image service
6. Kalshi public open-market data for `KXHIGHTLV`

## Install

```powershell
python -m pip install -e .
python -m pytest
```

Expected result for v0.14:

```text
51 passed
```

## Carry your trained model forward

v0.13 does not contain your locally trained `.joblib` model files. Copy the entire folder from v0.12:

```text
data\model\
```

into the same location in v0.13.

Optional: also copy your existing live history if you want today's progression to continue without restarting:

```text
data\live\klas_live_history.csv
```

## Run the live dashboard

```powershell
python scripts\live_update.py
```

Then open:

```text
docs\index.html
```

The terminal now prints the basic model line plus a weather-intelligence line, for example:

```text
KLAS 91.4F | NWS 96.0F | model 95.0F | risk MEDIUM | status NO STRONG EDGE
weather intel: PoP 30% | thunder True | AFD MEDIUM | radar LOW
```

If one of the new external sources is temporarily unavailable, the live update continues with a warning and the dashboard marks that source unavailable rather than crashing the entire model.

## Model integrity

Historical model workflow remains:

- train: 2022–2024
- tune/validation: 2025
- untouched test: 2026

The 2026 year-split test showed the model improving over the same-day NWS morning baseline as KLAS observations accumulated, reaching about **0.70°F MAE at the 2 PM checkpoint** in that test set. That historical result should continue to be treated as a research result, not a guarantee of future performance.

## v0.13.1 live-data resilience

The live dashboard now retries transient IEM ASOS failures and automatically falls back to the official AviationWeather.gov METAR API for recent KLAS observations. Historical training/backfill data remains IEM-based for consistency. The live updater prints the observation source used on each refresh.

## v0.14 — automatic GitHub Pages deployment

v0.14 adds the production-style hourly deployment path:

- scheduled GitHub Actions refresh at about `:05` after each hour in `America/Los_Angeles`
- persistent `data/live` hourly history committed by the workflow
- automatic GitHub Pages artifact upload and deployment
- concurrency protection so hourly jobs do not write the history simultaneously
- deployment preflight that refuses to run if the trained `h08`–`h18` model bundles are missing
- terminal AFD status now uses the same airport-adjusted AFD risk shown on the dashboard
- radar card now states the nearest coarse-ring echo distance (or that none is detected within 50 miles)

See `DEPLOY_TO_GITHUB.md` for the one-time setup.

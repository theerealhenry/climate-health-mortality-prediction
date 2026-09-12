# Data Dictionary

Merged reference from `data_dictionary.csv` (core competition columns) and
`downloaded_climate_features_data_dictionary.csv` (downloaded climate/environmental
features), both in `data/raw/`. See `docs/PROJECT_BLUEPRINT.md` §0.1 for verified
data-forensics findings that qualify several of these (e.g. `hot_days_30d` is
confirmed constant and should be dropped).

## Core columns (Train.csv / Test.csv)

| Column | Description |
|---|---|
| ID | Unique identifier for each mortality record |
| zone | Area classification (Rural or Peri_urban) |
| gender | Gender of the individual |
| deathdate | Date of death |
| age | Age of the individual at time of death |
| avg_temperature | Average temperature at the location and time of record |
| max_temperature | Maximum temperature at the location and time of record |
| min_temperature | Minimum temperature at the location and time of record |
| precipitation | Total precipitation at the location and time of record |
| latitude | Latitude coordinate of the location |
| longitude | Longitude coordinate of the location |
| location | Name of the location (village/area, inconsistent depth — see §0.1) |
| is_climate_sensitive | Binary target (Train only) |

## Downloaded climate features (climate_features.csv)

| Column | Description | Unit | Source |
|---|---|---|---|
| ID | Merge key back to core data | string/int | — |
| rain_sum_7d / 30d / 90d | Rainfall accumulated over N days before death date | mm | CHIRPS Daily |
| rain_days_30d | Days with >1mm rainfall in prior 30 days | days | CHIRPS Daily |
| max_daily_rain_30d | Highest single-day rainfall in prior 30 days | mm | CHIRPS Daily |
| tavg_7d / 30d / 90d | Mean 2m air temperature over N days | °C | ERA5-Land |
| tmax_30d / tmin_30d | Highest/lowest daily max/min temp in prior 30 days | °C | ERA5-Land |
| hot_days_30d | Days >35°C in prior 30 days — **confirmed constant (0), drop** | days | ERA5-Land |
| temp_range_mean_30d | Mean daily max−min temp range | °C | ERA5-Land |
| ndvi_30d / 90d | Median NDVI over N days | index (-1 to 1) | MODIS MOD13Q1 |
| elevation | Ground elevation | m | SRTM |
| slope | Terrain slope | degrees | Derived from SRTM |

## Stage 3 data forensics — findings (see `notebooks/00_data_forensics.ipynb` for full detail)

Full forensic audit run 2026-09-02 against the real raw files, cross-validated where
relevant (not single-fit). **Overall verdict: clean** — no leak, no near-deterministic
proxy, no circularity. One confirmed, actionable risk (twin records) is a validation-design
concern, not a forensics failure.

- **No leak / no proxy (the headline finding):** best single feature (`age`) reaches only
  0.734 cross-validated AUC; best two-feature combination (`age` + `max_daily_rain_30d`)
  reaches 0.764. Far below the ~0.95–0.99 range that would indicate a mechanically
  derivable target.
- **Twin-record risk, confirmed and quantified:** 66 groups (combined train+test) share
  identical climate features via `(location, deathdate)` (72 via the finer
  `(latitude, longitude, deathdate)`); 49/55 of those groups sit entirely within train.
  Within-train twin label agreement (57–58%) mildly exceeds the independent-assignment
  baseline (54.5%) — real spatial/temporal clustering in the target, not noise. **Stage 5's
  validation design must guard against this as a separate constraint from geographic
  generalization**, not assume grouped-CV alone covers it.
- **`age` is non-monotonic:** 0.836 (0–5, 55.8% of train) → dips to 0.367 (13–18, n=49) →
  rebounds to ~0.51–0.53 (19–45) → declines to 0.296 (60+). A recognizable
  young-child/elderly vulnerability pattern, not an artifact. Action: test an `is_under_5`
  flag and non-linear age treatment in Phase 3.
- **Rolling-window climate aggregates outperform same-day readings:** raw
  same-day temperature/precipitation columns are all near-chance (AUC 0.49–0.52); the
  30/90-day rolling aggregates — especially `rain_sum_90d`, `tavg_90d`, `ndvi_30d`,
  `ndvi_90d` — show meaningfully more target-rate spread. Prioritize these in Phase 3.
  `hot_days_30d` confirmed constant — drop.
- **`slope` has a genuinely different train/test distribution shape** (train std 1.167 vs.
  test std 0.065 — not just a mean shift). Not a leak (all merges are ID-based, never
  position-based), but flag it as a likely top driver in Stage 4's adversarial validation
  and treat any model reliance on it with caution.
- **`elevation`/`slope` sit on a coarse 23-value grid** across 50 locations — worth
  accounting for when Phase 3 builds spatial clustering.
- **Row-order/ID audit clean:** `ID` is a genuine opaque hash; Train/Test are
  `deathdate`-sorted, `climate_features.csv` is not (explains a benign −0.178 row-position/
  target correlation that cannot leak, since every merge in this project joins on `ID`).

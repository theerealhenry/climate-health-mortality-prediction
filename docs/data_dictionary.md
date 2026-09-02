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

# Notebooks

Per the project blueprint's "notebooks investigate, `src/climate_health/` implements"
principle (Stage 1): these notebooks are the research record, not where production
logic lives. Once a notebook establishes an answer, the implementation moves to
`src/climate_health/` and the notebook is updated to call it.

Planned sequence (populated stage by stage, not all at once):

- `00_data_forensics.ipynb` — Stage 3
- `01_eda.ipynb` — Stage 4
- `02_feature_engineering_exploration.ipynb` — Stages 6–9
- `03_climate_and_spatial_research.ipynb` — Stages 7–8
- `04_modeling_experiments.ipynb` — Stages 10–14
- `05_interpretability.ipynb` — Stage 21

`00`, `01`, and `05` are kept rendered (not stripped by pre-commit) since they're
meant to be read, not just re-run.

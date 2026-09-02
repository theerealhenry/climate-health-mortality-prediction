<div align="center">

# 🌍 Climate-Sensitive Mortality Prediction

### Detecting climate-linked patterns in public health mortality data

[![Python](https://img.shields.io/badge/python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Linting: ruff](https://img.shields.io/badge/linting-ruff-D7FF64?logo=ruff&logoColor=black)](https://github.com/astral-sh/ruff)
[![Experiment Tracking: MLflow](https://img.shields.io/badge/tracking-MLflow-0194E2?logo=mlflow&logoColor=white)](https://mlflow.org/)
[![Data Validation: Pandera](https://img.shields.io/badge/data%20validation-pandera-2C3E50)](https://pandera.readthedocs.io/)
[![Status](https://img.shields.io/badge/status-Phase%200%20%E2%80%94%20Foundation%20in%20progress-orange)](docs/PROJECT_BLUEPRINT.md)

*An end-to-end, production-grade machine learning pipeline I built for the Climate & Health Risk Prediction Challenge, engineered to the standard I'd want on any production ML system — not a notebook that happens to score well.*

[The Problem](#-the-problem) · [My Approach](#-my-approach--goals) · [Key Findings](#-key-data-driven-findings) · [Architecture](#%EF%B8%8F-architecture) · [Progress](#-project-progress) · [Getting Started](#-getting-started) · [Full Blueprint](docs/PROJECT_BLUEPRINT.md)

</div>

---

## 📌 Overview

Health outcomes are shaped by far more than biology — age, geography, living conditions, and environmental exposure all influence vulnerability to illness and mortality, particularly in low-resource settings where shifts in rainfall and temperature translate directly into elevated health risk. In this project I build a supervised machine learning system that predicts whether a recorded death falls into a **climate-sensitive category**, using demographic, geographic, and climate/environmental data.

I built it as two things at once, deliberately, on one shared engineering foundation:

1. **A competition entry** for the Climate & Health Risk Prediction Challenge, targeting first place.
2. **A senior-level ML engineering portfolio piece** — data validation, rigorous experiment design, MLflow-tracked experimentation, tested and typed production code, CI/CD, and a deployed, interactive model, documented to the standard I'd expect from a production ML system.

The full architectural reasoning, every design decision I made (including the ones I considered and rejected), and my stage-by-stage execution plan live in **[`docs/PROJECT_BLUEPRINT.md`](docs/PROJECT_BLUEPRINT.md)** — the single source of truth for this repository. This README is the front door; the blueprint is where I show my full engineering judgment.

## 🎯 The Problem

The challenge asks for a binary classifier that outputs, for each mortality record:

- `TargetF1` — a binary prediction (climate-sensitive or not), evaluated at a **fixed 0.5 threshold — threshold tuning is explicitly forbidden by the competition rules.**
- `TargetRAUC` — the predicted probability of the positive class.

The leaderboard score is a weighted blend of both:

$$\text{Final Score} = 0.60 \times \text{F1-Score} + 0.40 \times \text{ROC-AUC}$$

That fixed-threshold rule is more consequential than it first appears: it means I can't lean on the usual competition trick of hunting for the best decision threshold post hoc. Winning F1 at a threshold I don't control means my model's probabilities have to be genuinely well-calibrated around 0.5 — which shapes my model selection, calibration strategy, and validation design throughout this repository (see [`docs/PROJECT_BLUEPRINT.md`](docs/PROJECT_BLUEPRINT.md), Stage 15).

**Competition window:** 18 Aug 2026 – 19 Oct 2026 · **Prize:** 🥇 $500 (1st place) — my target.

## 🧭 My Approach & Goals

My guiding principle is that winning the competition and building a credible portfolio piece are not in tension — I pursue both on one shared foundation (data validation, feature pipelines, cross-validation, MLflow, tests) rather than building a rigorous pipeline for the competition and bolting on "portfolio polish" afterward. Concretely, this means:

- **Evidence over assumption.** I verify every structural claim about the data — geographic overlap between train and test, leakage risk, near-determinism of the target — against the raw files before letting it drive a design decision. See [Key Data-Driven Findings](#-key-data-driven-findings) below.
- **A validation strategy that respects the real generalization problem**, not a convenient one. Random K-fold looks impressive locally and lies about the private leaderboard, so I use a multi-tier, geography-aware validation architecture instead (details in my blueprint, Phase 2).
- **Disciplined experimentation.** Every submission I make answers a stated question (an `S`/`F`/`M`/`E`/`C` series scheme — sanity, feature, model, ensemble, champion) and gets logged with its hypothesis and result — I don't submit just because I have a new model.
- **Production standards from day one.** Schema validation, tested feature transformers, experiment tracking, and CI are first-class work for me starting in Phase 0, not deferred to "if there's time."
- **Honest scope.** I calibrate my claims about real-world impact to what a ~3,100-row, single-country, single-competition dataset can actually support — see [Responsible ML & Limitations](#%EF%B8%8F-responsible-ml--limitations).

## 🔍 Key Data-Driven Findings

These aren't assumptions — I verified them directly against `Train.csv`, `Test.csv`, and `climate_features.csv` during my Stage 1 data forensics pass, and they actively shape my modeling strategy:

| Finding | Value | Why it matters |
|---|---|---|
| Train/test geographic overlap | **0 of 43** training coordinates appear in test | Rules out naive random validation; drives my geography-aware, multi-tier CV strategy |
| Train/test location-name overlap | **1 of 39** locations shared | Confirms the coordinate finding through an independent signal |
| Non-independent "twin" records | **~55 groups** share identical place + death date | A second, distinct leakage mechanism, which I guard against separately from geographic generalization |
| Strongest single predictor | `age`, r ≈ **−0.44** | Younger age strongly predicts climate-sensitive death — consistent with real epidemiology (infant/child vulnerability to climate-linked illness) |
| Target balance | **65% / 35%** | Moderate imbalance, informs my class-weighting and calibration choices |
| Near-deterministic proxy for target | **None found** | Age alone reaches ≈0.74 AUC — strong signal, not a data leak (would be ≈0.99 if the target were mechanically derivable) |

Full methodology and my complete forensics writeup: [`docs/PROJECT_BLUEPRINT.md §0.1`](docs/PROJECT_BLUEPRINT.md).

## 🏗️ Architecture

I run two tracks — competition research and production engineering — on one shared foundation, converging on a single champion model:

```
                       SHARED FOUNDATION
        data validation → features → CV → tests → MLflow
                             │
        ┌────────────────────┴────────────────────┐
        ▼                                          ▼
  COMPETITION RESEARCH                     PRODUCTION SYSTEM
  forensics · EDA · climate & spatial      API · UI · CI/CD · Docker ·
  feature research · model zoo ·           docs · model card
  tuning · ensembling · calibration        (built once there's a
        │                                   stable artifact to serve)
        └────────────────────┬────────────────────┘
                              ▼
                       CHAMPION MODEL
                              │
                 ┌────────────┴────────────┐
                 ▼                         ▼
             FastAPI                  Streamlit
           (inference)              (interactive demo)
```

Nine phases, 24 stages, each with a stated rationale and deliverable — see my [full blueprint](docs/PROJECT_BLUEPRINT.md) for the complete breakdown, including my validation hierarchy, the climate-feature provenance/leakage-timing table, and my champion-model promotion gates.

## 📁 Repository Structure

```
├── .github/workflows/     # CI (lint/test on every push) + CD (build/deploy on tag)
├── configs/                # YAML configs: paths, model hyperparameters, CV settings
├── data/
│   ├── raw/                 # Train.csv, Test.csv, climate_features.csv (versioned as-is)
│   ├── interim/              # Merged, type-cast, pre-feature-engineering
│   ├── processed/             # Final model-ready feature matrices
│   └── external/               # Additional downloaded climate/geo data
├── docs/
│   ├── PROJECT_BLUEPRINT.md    # Single source of truth — full architecture & rationale
│   ├── data_dictionary.md
│   ├── model_card.md            # Added at Stage 22
│   └── retrospective.md          # Added at Stage 24, post competition close
├── notebooks/               # My research record — see notebooks/README.md for the
│                             # "notebooks investigate, src/ implements" principle
├── reference/                # Untouched copy of the official competition starter notebook
├── src/climate_health/
│   ├── data/                  # Loading, pandera schema validation
│   ├── features/                # sklearn-compatible transformers (temporal, spatial, climate)
│   ├── models/                   # Training, calibration, ensembling, inference
│   ├── evaluation/                # CV strategy, metrics, experiment/submission logging
│   └── utils/
├── app/
│   ├── api/                    # FastAPI inference service (Phase 7)
│   └── streamlit_app/            # Interactive demo UI (Phase 7)
├── tests/                     # pytest — schemas, transformers, CV splitter, environment smoke test
├── submissions/                 # Every submission I make, logged with its hypothesis and result
├── pyproject.toml                # Dependencies (pinned ranges) + tool config
├── environment.yml                # Conda bootstrap (Python 3.11 + pip)
└── requirements-lock.txt           # Exact resolved versions, generated post-install
```

## 🧰 Tech Stack

| Concern | Tools |
|---|---|
| Data validation | Pandera |
| Modeling | scikit-learn · LightGBM · XGBoost · CatBoost |
| Spatial features | k-means clustering on coordinates + climate normals |
| Hyperparameter tuning | Optuna |
| Experiment tracking | MLflow (SQLite-backed) |
| Interpretability | SHAP |
| Testing | pytest |
| CI/CD | GitHub Actions |
| Serving | FastAPI |
| Demo UI | Streamlit |
| Environment | conda (Python 3.11) + pip, pinned via `pyproject.toml` |

## 🧪 Methodology Highlights

- **Multi-tier cross-validation** — I report standard stratified K-fold, geography-grouped K-fold, and a test-like geographic holdout together for every candidate model, not just one convenient CV number, because the confirmed zero coordinate overlap between train and test means a single random split would systematically overstate performance.
- **Two leakage mechanisms, guarded separately** — I treat location-generalization leakage (unseen test locations) and non-independence leakage (same-day, same-place "twin" records) as mechanically distinct risks, not conflated into one.
- **Calibration checked, not assumed** — because the F1 threshold is fixed at 0.5 by competition rule, I inspect the model's raw reliability diagram before deciding whether Platt/isotonic calibration is even needed, rather than applying it reflexively.
- **A champion-model gate** — I don't call any candidate final until it passes reproducibility, leakage, and cross-fold stability checks (Phase 6 of my blueprint).

## 📊 Project Progress

*I update this after every stage I complete — see [`docs/PROJECT_BLUEPRINT.md`](docs/PROJECT_BLUEPRINT.md) for the full stage-by-stage detail behind each row.*

| Phase | Focus | Status |
|---|---|---|
| **0 — Foundation** | Repository, environment, data contracts | ✅ Complete — Stage 1 (repo & environment) and Stage 2 (data contracts) both done |
| 1 — Data Understanding | Forensics, EDA | 🟡 Up next |
| 2 — Validation Architecture | Multi-tier cross-validation | ⬜ Not started |
| 3 — Feature Research | Temporal, spatial, climate enrichment | ⬜ Not started |
| 4 — Baselines & Model Zoo | LightGBM · XGBoost · CatBoost | ⬜ Not started |
| 5 — Optimization & Ensembling | Optuna · stacking · calibration | ⬜ Not started |
| 6 — Competition Discipline | Submission registry, champion gate | ⬜ Not started |
| 7 — Production Engineering | Testing, CI/CD, deployment | ⬜ Not started |
| 8 — Responsible ML & Docs | SHAP, model card, retrospective | ⬜ Not started |

<details>
<summary><strong>Stage 1 details (complete)</strong></summary>

- Scaffolded the full repository skeleton per my blueprint
- Set up a conda environment (Python 3.11) + `pyproject.toml` dependency set, version-pinned for verified mutual compatibility
- Wrote an environment smoke test (`tests/test_environment_smoke.py`) — 10/10 passing: every core library imports cleanly and completes a real fit/predict/log round-trip (LightGBM, XGBoost, CatBoost, scikit-learn, SHAP, MLflow, Pandera)
- Standardized MLflow on a SQLite tracking backend after my smoke test caught MLflow 3.x deprecating the plain filesystem store — a real finding, not a hypothetical, and exactly what this stage is for
- Configured pre-commit hooks (ruff, black, nbstripout) to run against my project's own pinned tools rather than pre-commit's network-dependent hosted hook environments

</details>

<details>
<summary><strong>Stage 2 details (complete)</strong></summary>

- Wrote pandera `DataFrameSchema` contracts for every raw file (`src/climate_health/data/schemas.py`): `TRAIN_SCHEMA`, `TEST_SCHEMA`, `CLIMATE_FEATURES_SCHEMA`, `SAMPLE_SUBMISSION_SCHEMA`
- Contracts enforce column dtypes, value ranges (e.g. latitude/longitude bounded to Uganda's envelope), categorical membership, ID uniqueness and pattern (`ID_[8 hex chars]`), non-null constraints, and two cross-field consistency checks: `max_temperature >= avg_temperature >= min_temperature` and `rain_sum_90d >= rain_sum_30d >= rain_sum_7d`
- Schemas validate lazily (`lazy=True`) so a bad file reports every violation in one pass, not just the first
- Built schema-validated loaders (`src/climate_health/data/loaders.py`) — `load_train()`, `load_test()`, `load_climate_features()`, `load_sample_submission()`, `load_all()` — so nothing downstream ever calls `pd.read_csv` on a raw file directly
- Wired a CI-ready CLI gate (`src/climate_health/data/validate.py`, run via `make validate`) that validates every raw file and exits non-zero on any contract violation
- Verified all four real raw files validate cleanly against their contracts, and wrote 15 tests (`tests/test_schemas.py`) covering both the positive case and 11 deliberately corrupted negative cases (duplicate IDs, malformed ID patterns, broken temperature/rainfall ordering, invalid categories, out-of-range coordinates, null required fields, leaked target columns, missing files) — every corruption is confirmed caught, not just assumed to be

</details>

## 🚀 Getting Started

```bash
git clone https://github.com/theerealhenry/climate-health-mortality-prediction.git
cd climate-health-mortality-prediction

conda env create -f environment.yml
conda activate climate-health
pip install -e ".[all]"

pytest tests/test_environment_smoke.py -v   # verify the environment before doing anything else
```

## 📈 Results

*I'll populate this as I train and validate models (Phase 4 onward). Local cross-validation scores, public leaderboard scores, and — once the competition closes — the private leaderboard score will be reported here alongside the CV-vs-private gap, the strongest evidence for whether my validation strategy actually worked.*

## 🖥️ Live Demo

*Coming in Phase 7: a FastAPI inference endpoint behind a Streamlit UI, deployed as a single Hugging Face Space. I'll add the link here once it's live.*

## ⚖️ Responsible ML & Limitations

This model estimates whether a mortality record is likely to belong to the climate-sensitive category as defined by this competition's dataset. It's a research and decision-support demonstration — **not a clinical or individual mortality-risk tool.**

I built this to demonstrate a methodology genuinely used in public-health decision support: combining demographic, geographic, and environmental records to identify climate-sensitive mortality patterns, with longer-term potential to inform climate-health surveillance and resource planning. It doesn't itself save lives — it's a model I trained on ~3,100 records from a single competition dataset, covering one country over a 15-year span, with only 39 distinct training locations. I state those limitations plainly in my [model card](docs/model_card.md) (added at Stage 22), not glossed over.

## 🗺️ Roadmap

| Week | Focus |
|---|---|
| 1 | Foundation, data forensics, deep EDA, first baselines |
| 2 | Validation architecture locked in, temporal/demographic features |
| 3 | Spatial + climate feature research; first validated submission; go/no-go checkpoint vs. public leaderboard |
| 4 | Full model zoo, initial comparison table |
| 5 | Hyperparameter tuning, ensembling |
| 6 | Calibration, robustness, champion gate, final competition submissions |
| 7 (close: 19 Oct) | Buffer for last CV-guided submissions |
| Post-close | Deployment, interpretability, documentation, retrospective |

## 👤 About Me

**Henry Otsyula**
Data Scientist & Machine Learning Engineer

GitHub: [@theerealhenry](https://github.com/theerealhenry) · LinkedIn: [henry-otsyula-datascientist](https://www.linkedin.com/in/henry-otsyula-datascientist) · Email: [henryotsyula01@gmail.com](mailto:henryotsyula01@gmail.com)

## 🙏 Acknowledgments

- The competition organizers, for a dataset and problem framing grounded in a genuine public-health question
- **CHIRPS Daily** (rainfall), **ERA5-Land** (temperature/atmospheric), **MODIS MOD13Q1** (NDVI), and **SRTM** (elevation/slope) — the public climate and environmental data sources underlying the downloaded climate features I used

## 📄 License

MIT — see [`LICENSE`](LICENSE).

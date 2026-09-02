# Climate-Sensitive Mortality Prediction — Project Blueprint (v2)

**Single source of truth for the project.** Every phase below states what is done, why, what gets created, and the concrete deliverable. This document is versioned in the repo at `docs/PROJECT_BLUEPRINT.md` and updated in place as decisions evolve — treat changes to it like an ADR (architecture decision record) log, not a static plan. v2 supersedes v1 after two independent technical reviews; §0.2 records what changed and why, so the reasoning isn't lost.

Owner: Henry Otsyula. Competition: Climate & Health Risk Prediction Challenge. Opened 18 Aug 2026, closes 19 Oct 2026 (public leaderboard is a partial-test-set score during the competition; the info page states the full private evaluation is revealed at close — confirmed from the competition's own leaderboard page text, not assumed). Prize: 1st place $500.

## 0.1 Verified data forensics (run before writing this revision, not assumed)

v1 stated some structural claims about the data as fact without having executed the code to confirm them. Both reviews flagged this as the single most important gap — most sharply Review 2 point 1, which named it a blocking dependency on the whole validation strategy — so before revising anything else, the checks were actually run against the raw files. Results, which now anchor every downstream decision in this document:

- **Location overlap, confirmed:** 39 unique `location` strings in Train, 11 in Test, exactly **1** shared (`Nakigo I, Iganga, Uganda`). This is the number v1 asserted; it checks out.
- **Lat/lon overlap is even more stark than the location-string number suggested:** 43 unique `(latitude, longitude)` pairs in Train, 12 in Test, and **0 exact coordinate matches between Train and Test**. Every test coordinate is geographically novel relative to training. This is stronger evidence for grouped/spatial validation than the location-string overlap alone, and it rules out the "GroupKFold might be needlessly pessimistic" concern Review 1 raised — the geographic split is real and total, not a string-parsing artifact.
- **A second, distinct leakage mechanism exists and is non-trivial:** grouping Train by `(location, deathdate)` finds **49 groups with more than one record** (98 records total, i.e. ~3% of Train); grouping by the finer `(latitude, longitude, deathdate)` finds **55 such groups**. These are different people who died on the same day in the same place and therefore received identical climate-feature rows — they are not independent observations. Plain K-fold, even stratified, can split these "twin" records across train/validation and inflate apparent CV performance regardless of the location-generalization question. This confirms Review 2's point 6: it's a mechanically distinct risk from location generalization, and both need to be named and guarded against explicitly, not conflated into one.
- **No near-deterministic single-variable or two-variable proxy for the target was found.** Age-binned target rates range from 0.30 (60+) to 0.92 (0–5) — a real and strong pattern, not a leak (age legitimately drives climate-sensitive-cause mortality; there's no reason age should be in the data purely as a target proxy). Location-level target rates range from 0.33 to 0.91, but the extremes sit on tiny counts (n=3, n=11); no high-count location approaches 0 or 1. A quick univariate logistic-regression check on `age` alone gets ≈0.74 AUC — strong, but nowhere near ≈0.99, which is the level that would indicate the target was mechanically derived from a feature we can see. `hot_days_30d` is confirmed constant (all zero) across all 4,176 rows — dead column, drop it. `ID` is an opaque hash with no encoded ordering; row order itself is sorted by `deathdate`, which is a minor reproducibility note (don't rely on row order as a feature; do note it if any date-based leakage check needs to account for it), not a leak.
- **Net effect on the validation-strategy debate (Review 1 §2–3 vs. the v1 "one fixed GroupKFold" line):** the coordinate-level evidence makes grouped/geographic validation clearly correct as the *primary* decision signal, not merely "provisional pending a check" — the check has been run. What Review 1 got right is that reporting only one grouped-CV number is still under-informative at 39 groups; that part is adopted below as a validation hierarchy, not because the original grouping decision was in doubt, but because its uncertainty needs to be visible.

## 0.2 How the two reviews were adjudicated

Both reviews were read in full and checked against the actual data where they made a checkable claim (§0.1 above covers the empirical ones). What follows is the disposition of every substantive point, so the reasoning is auditable later rather than silently baked in.

**Adopted as-is or with light scoping — both reviews, or independently verified:**
Data forensics as an explicit stage before EDA (Review 1 §1); a validation *hierarchy* rather than a single CV number, sized to what 39 location-groups can actually support — mean/std/min/max across folds, repeated group splits (Review 1 §2–3); the adversarial-validation re-interpretation — a high train/test-distinguishability AUC identifies *where* the distributions differ, not features to blindly drop, since some of those (lat/lon, climate normals) may be exactly what lets the model generalize (Review 1 §5); climate enrichment promoted to its own phase with an explicit provenance/leakage-timing table per feature (Review 1 §6–8, Review 2's point about prediction-time information boundaries); the location string-parsing plan replaced with spatial/climate clustering as the primary geographic representation, since it's demonstrably more principled given 43 train coordinates and zero train/test coordinate overlap (Review 2 point 5, which subsumes and sharpens Review 1 §10); the rainfall-acceleration feature dropped for the arithmetic reason given (7×4=28≠30) plus its arbitrary interpretation, replaced with normalized/ratio features (Review 1 §9); OOF-only, Bayesian-smoothed spatial/zone target encoding as an experiment, gated by the leakage checks above (Review 1 §11); a broader model zoo used specifically for ensemble diversity, not for its own sake (Review 1 §13); a more rigorous ensembling stage — weighted/rank averaging, OOF stacking, explicit prediction-correlation/diversity analysis (Review 1 §14); the CatBoost/pipeline contradiction — a dual-path encoder so CatBoost (and XGBoost with `enable_categorical=True`) actually receives raw categoricals instead of a pre-one-hot matrix, which is the whole stated reason for including it (Review 2 point 3, a real bug in v1's own logic caught by the reviewer, not a stylistic preference); calibration ordering — tune first, but explicitly check the pre-calibration reliability diagram before trusting that Optuna's objective and the post-calibration submission are aligned (Review 2 point 4); an explicit univariate leakage/sanity table in the EDA stage, now already produced once in §0.1 and to be expanded per-feature in the notebook (Review 2 point 8); an explicit `git tag` at the exact submitted model, with interpretability/model-card work pinned to that artifact (Review 2 point 9); MLflow moved earlier, tracking from the first baseline rather than bolted on at the portfolio stage (Review 1 §22); two CI/CD workflows — CI on every push/PR, CD on tag (Review 1 §23); a lightweight experiment registry and a formal submission-numbering/leaderboard discipline, including an explicit "don't overfit the public LB" rule and a go/no-go checkpoint after the first grouped-CV-validated submission (Review 1 §18, §27–28; Review 2 point 11); champion-model promotion gates before anything is called final (Review 1 §29); full model governance/traceability — model, feature-pipeline, and training-data versions plus git SHA and MLflow run attached to every served artifact (Review 1 §25 — adopted in full, not scoped down, unlike the neighboring monitoring point; see Stage 22); the front-loaded, forensics-first timeline reordering (Review 1 §26); the deployment medical-disclaimer copy (Review 1 §20); the sharper portfolio-narrative wording (Review 1 §32, which improves on v1's own already-cautious version); the deployment scope hole — the demo cannot compute `tavg_30d`-style features for an arbitrary user-typed record, so it operates over the known record universe (select an existing location+date, perturb demographics) with live-API climate lookup explicitly deferred to "post-competition, if time permits" (Review 2 point 2, a real gap v1 would have hit mid-build); single deployment host instead of two, since splitting FastAPI and Streamlit across two free-tier providers is real operational risk for zero portfolio upside on a solo deadline project (Review 2 point 10).

**Adopted in modified/scoped-down form — good ideas, wrong altitude for a 7-week solo project:**
Review 1's 80-substage, 15-phase mega-architecture is a legitimate research program, not a plan a solo developer executes in 7 weeks alongside a day job. The underlying ideas are almost all sound and are folded into this document, but organized into a workable number of phases rather than reproduced at that granularity. Mixture-of-experts (Review 1 §15), spatial nearest-neighbor/residual modeling (Review 1 §12), a GAM/spline explainability model (Review 1 §13), and a full 24/7 input/prediction drift-monitoring service (Review 1 §24) are kept as explicitly-labeled **stretch items** — attempted only if the champion model, ensembling, and core deployment are already done with time remaining. Listing them as required core work would create exactly the kind of plan that looks impressive on paper and ships nothing; listing them as scoped stretch goals keeps the ambition visible without gating the deadline on it. The "shared foundation, not sequential tracks" reframing (Review 1 §21, §31) is adopted conceptually — data/feature/validation code and MLflow/tests are genuinely shared and start immediately — but heavy deployment/monitoring engineering still substantively lands after the champion model is locked (end of Phase 6, not v1's old "stage 9" numbering), because that is when there's a stable artifact worth deploying; that's not a contradiction of the reframing, it's what the reframing looks like once time-boxed.

**Rejected or corrected:**
Review 2's concern that the private-leaderboard mechanism might not exist (point 7) is resolved, not left open: the competition's own leaderboard page states scores reflect only a portion of the test set until close, with a separate "Reveal" date — that is a public/private split by definition, so stage 15's CV-vs-private-score comparison stands as planned. Uganda rainy-season date flags (flagged by Review 1 §9 as a premature hard-coded assumption) are kept, but downgraded from "implement directly" to "derive candidate date ranges from public agro-climatic references, then validate them against the observed month-level target-rate pattern in the EDA before the pipeline depends on them" — the caution was right, outright removal would have thrown away a legitimate domain-knowledge feature.

## Two tracks, one shared foundation

The project has two objectives with different deadlines — win the competition (hard deadline 19 Oct 2026) and produce a senior-level portfolio piece (no hard deadline) — sharing one repository and one engineered core, rather than a portfolio layer bolted on after the fact:

```
                    SHARED FOUNDATION
        data validation → features → CV → tests
        config management → git discipline → MLflow
                          │
        ┌─────────────────┴─────────────────┐
        ▼                                   ▼
  COMPETITION RESEARCH                PRODUCTION SYSTEM
  forensics, EDA, climate/spatial     API, UI, CI/CD, Docker,
  research, model zoo, tuning,        docs, model card,
  ensembling, calibration             (built once there's a
        │                              stable artifact to serve)
        └─────────────────┬─────────────────┘
                           ▼
                    CHAMPION MODEL
```

MLflow, tests, and config discipline start at phase 2, not as a late addition — by the time a champion model exists, "add MLflow" is not a task, it already happened. Heavy deployment/monitoring work substantively begins once the champion model is locked (end of phase 6), because that is the first point there's a stable artifact worth serving — this is pragmatic sequencing under a real deadline, not a contradiction of the shared-foundation principle.

On portfolio framing: this project demonstrates a methodology genuinely used in public-health decision support — combining demographic, geographic, and environmental records to identify climate-sensitive mortality patterns, with longer-term potential to support climate-health surveillance and resource planning. It does not itself save lives; it's a model trained on ~3,100 records from one competition dataset. Claim what's true and impressive (rigorous methodology, a real public-health-relevant problem, production-grade engineering) rather than an impact claim the project can't substantiate — a hiring manager reads the honest version as more senior, not less.

## Phase 0 — Foundation

### Stage 1: Repository & environment

**What's done:** Git repo with a standard senior-level Python ML layout, dependency management, and tooling config, before any modeling code exists.

**Deliverables:**

```
climate-health-risk-prediction/
├── .github/workflows/          # ci.yml (every push/PR) + release.yml (on tag)
├── configs/                    # YAML: paths, model hyperparams, CV settings
├── data/
│   ├── raw/                    # Train.csv, Test.csv, climate_features.csv (untouched)
│   ├── interim/                # merged, type-cast, pre-feature-engineering
│   ├── processed/               # final model-ready feature matrices
│   └── external/                # any additional downloaded climate/geo data
├── docs/
│   ├── PROJECT_BLUEPRINT.md    # this file
│   ├── model_card.md
│   ├── retrospective.md
│   ├── data_dictionary.md
│   └── experiment_registry.md  # human-readable hypothesis log; the machine-readable
│                                # companion is submissions/leaderboard_log.csv — both
│                                # exist, they are not alternatives (see Phase 6)
├── notebooks/
│   ├── 00_data_forensics.ipynb
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering_exploration.ipynb
│   ├── 03_climate_and_spatial_research.ipynb
│   ├── 04_modeling_experiments.ipynb
│   └── 05_interpretability.ipynb
├── src/climate_health/
│   ├── data/                   # loading, schema validation
│   ├── features/                # sklearn-compatible transformers (temporal, spatial, climate)
│   ├── models/                  # training, calibration, ensembling, inference
│   ├── evaluation/               # CV strategy, metrics, experiment/submission logging
│   └── utils/
├── app/
│   ├── api/                     # FastAPI inference service
│   └── streamlit_app/            # interactive demo UI (same container/Space as api)
├── tests/
├── mlruns/                        # MLflow local tracking store (gitignored)
├── models/                        # serialized artifacts (gitignored)
├── submissions/                    # every submission.csv + its metadata row, kept for audit
├── Dockerfile
├── pyproject.toml
├── Makefile                        # make forensics / eda / train / test / serve
├── .pre-commit-config.yaml
├── .gitignore
└── README.md
```

Python 3.11, `pyproject.toml`, `uv` (fast, modern, a good current signal) or `venv`. Pre-commit runs `ruff`, `black`, `nbstripout` (strips outputs before commit for every notebook except `00_data_forensics.ipynb`, `01_eda.ipynb`, and `05_interpretability.ipynb` — the three narrative/findings notebooks meant to be read rendered, not re-run, by a portfolio reviewer; `02`–`04` are working/exploration notebooks and stay stripped).

**Recorded decision:** raw data (~1.7 MB total) is committed directly to `data/raw/` rather than via DVC — DVC is a real skill worth having but adds setup overhead with no benefit at this size; note the considered-and-rejected choice in the README.

**Recorded decision (found by the Stage 1 environment smoke test, not anticipated in planning):** MLflow 3.x has put the plain filesystem tracking backend (`./mlruns`) into maintenance mode — `mlflow.start_run()` against a `file://` URI now raises unless explicitly opted back into a deprecated path. Rather than opt out of a backend MLflow itself is deprecating, the project standardizes on a SQLite tracking store instead: `mlflow.set_tracking_uri("sqlite:///mlflow.db")` at the repo root, used consistently from Stage 10 onward. `mlflow.db` and `mlflow_artifacts/` are gitignored, matching `mlruns/` in intent. This is exactly the kind of thing Stage 1's smoke test exists to catch before it costs time mid-pipeline — recorded here rather than only in a commit message so the reasoning survives.

**A cross-cutting engineering principle, stated once here because it governs every later phase (Review 1 point 19):** notebooks investigate, `src/climate_health/` implements. A notebook is where "does a 30-day rainfall anomaly help" gets explored and answered; once the answer is yes, the actual feature computation is written as a tested function/transformer in `src/climate_health/features/`, and the notebook is updated to call that function rather than keep its own copy of the logic. Training, calibration, and inference (Stages 13, 15, 19) never re-implement anything a notebook already worked out — they import it. This is what keeps the notebooks in `notebooks/` honest as a research record instead of drifting out of sync with what actually ships.

### Stage 2: Data contracts

**What's done:** `pandera` schemas for `Train.csv`, `Test.csv`, `climate_features.csv` — column presence, dtypes, category domains for `zone`/`gender`, plausible ranges, target domain `{0,1}` — run on every raw load, failing loudly on violation.

**Deliverables:** `src/climate_health/data/schemas.py`; a `make validate` target; `tests/test_schemas.py`.

## Phase 1 — Data understanding

### Stage 3: Data forensics & target investigation

**What's done:** Before asking "which model predicts the target," ask "how does this target appear to have been generated, and is anything in the data secretly encoding it." §0.1 above is the first pass of this and is not hypothetical — it already found the location/coordinate overlap numbers, the twin-record counts, and confirmed no near-deterministic proxy exists. The notebook expands that into the full checklist: target prevalence by every individual variable and by the most plausible 2–3 way combinations (age×zone, age×location-cluster); duplicate/near-duplicate demographic-geographic profiles; whether any provided variable could only exist *because* the target is known (checked — no evidence of this); and a written note on the sorted-by-date row order (harmless, but worth stating so nobody later assumes row order is random).

**Why:** This is a data-generating-mechanism question, not a fitting question — if age+location nearly determined the target we'd be solving a fundamentally different (and easier, and less interesting) problem than "demographics + climate → target." Confirming it does not is itself a load-bearing finding for the rest of the plan.

**Deliverables:** `notebooks/00_data_forensics.ipynb`; a findings block mirrored into `docs/data_dictionary.md`; §0.1 of this document is the seed of that notebook.

### Stage 4: Exploratory data analysis

**What's done:** Univariate distributions for every feature; bivariate relationships against target using both linear correlation and mutual information (linear correlation already understated the climate features in the v1 pass — MI is likely to tell a richer story); a geospatial view (lat/lon colored by target and zone); a temporal view (target rate by month/year, checked against candidate Uganda rainy-season windows before those windows are hard-coded into features, per §0.2); and adversarial validation (train a classifier to separate Train rows from Test rows) — interpreted per §0.2's correction: a high AUC tells us *where* the distributions differ (very likely dominated by location/coordinates given the 0% coordinate overlap already confirmed), which features are carrying that signal, and which of those are worth keeping *because* they generalize (climate normals, elevation) versus ones that are pure location fingerprints and risk overfitting to the 39 training places.

Alongside this, a direct **train-vs-test distribution comparison** across every dimension that could plausibly differ — age, gender, zone, spatial cluster, each climate normal, and the temporal (month/year) spread — not just the adversarial classifier's overall verdict. This is Review 1's "Test-Likeness Validation Framework" proposal (§0.2), folded into this stage rather than built as a separate freestanding tool: a standalone framework would duplicate most of what the adversarial-validation feature-importance breakdown and these per-dimension comparisons already produce together, and the output — which dimensions Train and Test actually differ on — is exactly what Stage 5's Tier 3 holdout needs in order to mimic the real train/test relationship rather than an arbitrary geographic split.

**Deliverables:** `notebooks/01_eda.ipynb`, fully rendered and kept in git (not stripped by `nbstripout` — see Stage 1); the adversarial-validation AUC and its feature-importance breakdown, plus the per-dimension train-vs-test comparison table, both of which directly gate what Phase 3's spatial features are allowed to use and how Stage 5's Tier 3 holdout is constructed.

## Phase 2 — Validation architecture

### Stage 5: Multi-tier validation strategy

**What's done:** Given the confirmed geographic split (§0.1: zero coordinate overlap, near-zero location overlap) and confirmed non-independence among twin records, one CV number is not enough to trust at only 39 location groups. Three tiers, all computed for every candidate model, not just the final one:

| Tier | Scheme | Purpose |
|---|---|---|
| 1 — Standard | Stratified K-fold (grouped additionally by `(location, deathdate)` to guard against twin-record leakage even here) | Ordinary predictive capability; detects whether a change genuinely improves the classifier at all |
| 2 — Geographic | GroupKFold by spatial cluster (Phase 3), repeated with multiple random group-to-fold assignments given only ~39 groups, reporting mean/std/min/max across folds | Tests geographic generalization directly |
| 3 — Test-like holdout | A single held-out split constructed to resemble the actual train/test structure — locations absent from the "training" portion, and matched on whichever other dimensions Stage 4's train-vs-test comparison found actually differ (not just geography, if age/climate distributions turn out to differ too) | The closest local proxy to "what happens on genuinely unseen places," used sparingly (see Phase 6 gate) so it isn't itself optimized into uselessness |

Both leakage mechanisms are named and guarded separately, not conflated: **location-generalization leakage** (Test locations weren't seen in training) and **non-independence leakage** (twin records sharing a day+place aren't independent draws even within Train). Tier 2's grouping needs to key on both location cluster *and* date-proximity for records at the same coordinates to fully guard the second one.

Given the small number of independent geographic groups, a single unusual location can swing a 5-fold score noticeably — hence reporting spread (std, min/max), not just a mean, in every model-comparison table (Phase 5).

**Deliverables:** `src/climate_health/evaluation/cv.py` implementing all three tiers; every subsequent experiment reports all three, tabulated together.

## Phase 3 — Feature research

### Stage 6: Temporal & demographic features

**What's done:** Date decomposition (day-of-year, month, cyclical sin/cos, year trend), candidate Uganda rainy-season flags derived from public agro-climatic references and *validated against the observed month-level target-rate pattern from Stage 4 before use* (not hard-coded blind); age transforms matched to public-health age bands (under-5, 5–17, 18–59, 60+, plus a continuous log/sqrt version and an `is_under5` flag given age's dominance in the forensics pass).

**Deliverables:** `src/climate_health/features/temporal.py`, `demographic.py`; unit tests for edge cases (leap years, boundary ages).

### Stage 7: Spatial features (redesigned from v1)

**What's done:** String-parsing `location` into district/region tokens is kept only as a *fallback/secondary* feature (documented as brittle given the 3–8 token depth variance found in v1's own EDA), not the primary geographic representation. The primary representation is **spatial clustering**: k-means (5–8 clusters, tuned) over the 43 training coordinates jointly with location-level climate normals (`tavg_30d`, `rain_sum_90d`, `elevation`), so an unseen test coordinate is assigned to its nearest cluster centroid by feature similarity — something a parsed district string cannot do once a test location's district is itself unseen, which the coordinate-overlap finding in §0.1 shows is the normal case here. Raw lat/lon and simple transforms (lat², lon², lat×lon) are included directly as features alongside the cluster assignment.

**Deliverables:** `src/climate_health/features/spatial.py` (clustering transformer, fit on train coordinates + climate normals, `.transform()` assigns nearest cluster to any coordinate including unseen test ones); `notebooks/03_climate_and_spatial_research.ipynb` documenting cluster count selection and what separates the clusters.

### Stage 8: Climate feature engineering & enrichment (promoted to its own research track)

**What's done:** Treated as a real research phase, not a feature-engineering afterthought, in four tracks:

| Track | Content |
|---|---|
| A — Provided | Establish the value of `climate_features.csv` as delivered (drop the confirmed-constant `hot_days_30d`); derive exposure windows already implicit in the columns (acute: 7d, medium: 30d, chronic: 90d) as normalized/per-day and ratio features (`rain_sum_30d / 30`, `rain_sum_7d / rain_sum_30d`) rather than the arithmetically-wrong `rain_sum_30d − rain_sum_7d*4` from v1 |
| B — ERA5-style temperature | Anomalies relative to location-level climate normals, extreme-heat-day counts recomputed from `tmax_30d`/`tmin_30d` if usable (the provided `hot_days_30d` is dead, but the underlying temperature columns may still support a working version) |
| C — CHIRPS-style rainfall | Dry-spell indicators, rainfall anomaly vs. climate normal, seasonal accumulation aligned to the validated rainy-season windows from Stage 6 |
| D — NDVI/vegetation (stretch) | NDVI anomaly and short-term trend, only if Track A–C are exhausted with time remaining |

Every derived feature is entered into a **provenance/leakage-timing table** before use:

| Feature | Reference period | Available at prediction time? | Risk |
|---|---|---|---|
| `tavg_30d` | T−30 → T | Yes (as provided) | Low |
| `rain_sum_90d` | T−90 → T | Yes (as provided) | Low |
| `ndvi_30d` | T−30 (satellite composite, may lag) | Needs verification | Medium |

The distinction that matters: historical climate data availability in this competition dataset is not automatically the same as prediction-time availability in a real deployment. For the competition itself, using the provided historical windows is fine and expected. For the Phase 7 deployment demo, this table becomes the enforced boundary — the served model must only use information that would genuinely be available at prediction time, and that constraint is exactly why Phase 7's demo is scoped to known records rather than free-text new ones (see Stage 19).

Downloading further external sources (live ERA5/CHIRPS via a public API) is explicitly a **stretch item**, attempted only after Tracks A–C are exhausted and only if it doesn't threaten the submission timeline — it's real, valuable work, but it's its own mini-project and shouldn't eat competition weeks.

**Deliverables:** `src/climate_health/features/climate.py`; the provenance table checked into `docs/`; `notebooks/03_climate_and_spatial_research.ipynb` (shared with Stage 7).

### Stage 9: Interaction & encoded features

**What's done:** Interaction terms motivated by the EDA/forensics findings (`age × tavg_30d`, `spatial_cluster × zone`, climate-anomaly × age-band); and, gated strictly behind the leakage checks above, **out-of-fold, Bayesian-smoothed target encoding** for `spatial_cluster`, `region`, and `zone` (never raw `location`, which is too fine-grained relative to the group count to encode safely) — computed only within each CV fold's training portion, never leaking validation-fold targets into their own encoding.

**Deliverables:** `src/climate_health/features/interactions.py`, `encoding.py`; leakage unit tests verifying OOF encoding never sees its own fold's labels.

## Phase 4 — Baselines & model zoo

### Stage 10: Baselines

**What's done:** The starter notebook's Logistic Regression, reproduced inside the new pipeline and rescored under all three validation tiers (expect it to look worse under Tier 2/3 than the starter's plain random split suggested — that gap, once observed, is itself the evidence that Phase 2's validation redesign was necessary, and it's worth stating plainly rather than just asserting it as v1 did). A majority-class baseline and a default-hyperparameter LightGBM are added as the low and high anchors of the baseline scorecard.

**Deliverables:** Baseline scorecard (see Stage 12 table format) logged to MLflow from this stage onward — not deferred to a later "add tracking" step.

### Stage 11: Model zoo

**What's done:** LightGBM, XGBoost, and CatBoost as the primary candidates, with a corrected pipeline: features branch after engineering into one path that preserves categorical dtypes (`zone`, `gender`, `spatial_cluster`, `region`) for CatBoost and XGBoost's native categorical handling, and a second path that one-hot/target-encodes for Logistic Regression — so the stated reason for including CatBoost ("handles categoricals natively") is actually true of what gets fed to it, which it wasn't in v1's single shared-encoder design. HistGradientBoosting, ExtraTrees, and plain Logistic Regression are added specifically as **ensemble-diversity candidates** — not because any is expected to win outright, but because a model with different error structure than the GBMs can still improve a blend even at a lower standalone score.

**Deliverables:** `src/climate_health/models/` with a shared training interface across all candidates; `notebooks/04_modeling_experiments.ipynb`.

### Stage 12: Model comparison table

**What's done:** Every candidate reported across all three validation tiers with spread, not a single number:

| Model | Tier 1 (Random) | Tier 2 (Geo, mean±std) | Tier 3 (Test-like) | Competition score |
|---|---|---|---|---|
| Majority baseline | — | — | — | — |
| Logistic Regression | | | | |
| LightGBM (default) | | | | |
| LightGBM (tuned) | | | | |
| XGBoost (tuned) | | | | |
| CatBoost (tuned, native categoricals) | | | | |
| Ensemble | | | | |

The HistGradientBoosting/ExtraTrees/diversity candidates from Stage 11 are logged to MLflow and the registry like everything else, but only enter this headline table if they end up in the Stage 14 ensemble — their job is decorrelated errors for blending, not a standalone leaderboard row.

**Deliverables:** This table maintained live in `docs/experiment_registry.md` and mirrored in MLflow's comparison view.

## Phase 5 — Optimization & ensembling

### Stage 13: Hyperparameter tuning

**What's done:** Optuna tuning the strongest 1–2 candidates, objective = the literal competition score (0.60×F1@0.5 + 0.40×AUC) inside the Tier-2 geographic CV loop, with CV variance monitored alongside the mean so a config that wins on average but is unstable across folds isn't blindly promoted. A modest trial budget, explicitly because repeatedly optimizing against ~39 geographic groups risks overfitting the validation structure itself, not just the model — one fold combination is set aside as a **locked internal holdout**, used only once per candidate near the end of tuning, never as a per-trial signal.

**Deliverables:** `src/climate_health/models/tune.py`; persisted Optuna study; `configs/model_best.yaml`.

### Stage 14: Ensembling

**What's done:** Beyond a simple average: weighted averaging with OOF-optimized weights, rank averaging (useful if calibration differs across models), and OOF stacking with a simple meta-learner — always using out-of-fold predictions, never in-fold. Before combining, a prediction-correlation/disagreement analysis across candidates answers the actual question that matters: does model B make different mistakes from model A, not just "is model B individually good."

**Stretch (only if time remains after a working ensemble is locked):** a small "spatial expert vs. demographic expert vs. general GBM" gated blend, and/or a GBM-plus-spatial-residual model — genuinely interesting ideas, explicitly not required for the competition submission.

**Deliverables:** `src/climate_health/models/ensemble.py`; the diversity/correlation analysis in `notebooks/04_modeling_experiments.ipynb`.

### Stage 15: Calibration & threshold-aware finalization

**What's done:** Because `TargetF1` is fixed at 0.5 and threshold tuning is forbidden, this is not assumed to be automatically fine — it's checked. Plot the pre-calibration reliability diagram of the tuned champion model first. If it already sits close to the diagonal near p=0.5, calibration is documented as unnecessary and the tuning objective from Stage 13 is confirmed aligned with the final submission. If there's real miscalibration, fit both Platt and isotonic calibration on OOF predictions and tabulate all three (raw, Platt, isotonic) side by side on F1@0.5, AUC, competition score, Brier score, and the calibration curve itself, then pick the best-scoring row — not the first one tried. Note explicitly that ROC-AUC is invariant to monotonic recalibration, so any real gain from this step shows up in F1, not AUC, which is exactly why the comparison is worth doing carefully rather than skipping. Retrain on full training data, generate `submission.csv` through a single scripted entry point (never a manual notebook step), and **tag the exact commit** (`git tag competition-submission-final` at whichever point is actually the last submitted candidate) so every later interpretability/model-card artifact can be pinned to the model that was actually scored.

**Deliverables:** `src/climate_health/models/calibrate.py`, `predict.py`; the reliability-diagram check result documented either way ("checked, didn't matter" is a real portfolio artifact); `submissions/` archive with commit hashes; the `competition-submission-final` git tag.

## Phase 6 — Competition discipline

### Stage 16: Experiment registry & submission strategy

**What's done:** Every experiment recorded with hypothesis, data/feature/validation versions, model, CV scores across all three tiers, decision, and reasoning — not just a leaderboard log of numbers. Submissions are categorized by what question they're answering, never made simply because a new model exists:

| Series | Meaning | Example |
|---|---|---|
| S — sanity | Pipeline/schema/submission-format correctness, not a scoring attempt | `S-001`: confirm `submission.csv` round-trips through the scripted pipeline and matches `SampleSubmission.csv` exactly |
| F — feature experiment | Tests one feature family in isolation against the current best | `F-004`: does the rainfall-anomaly feature (Stage 8, Track C) improve Tier-2 CV over the current champion? |
| M — model experiment | Tests an algorithm/hyperparameter change | `M-011`: tuned CatBoost with native categoricals vs. tuned LightGBM |
| E — ensemble experiment | Tests a combination | `E-003`: OOF-weighted blend of `M-011` + `M-013` vs. either alone |
| C — champion candidate | Only strongest validated candidates, submitted to actually move the leaderboard | `C-002`: calibrated `E-003`, passed the Stage 17 gate |

Submission IDs follow `<series>-<3-digit-number>`, matched 1:1 to an entry in `docs/experiment_registry.md` giving the hypothesis, motivation, data/feature/validation code versions, baseline compared against, result, delta, and decision (keep/reject/modify) — e.g. an `F`-series entry states up front what it's testing and closes with why it was kept or dropped, not just a score. `submissions/leaderboard_log.csv` is the machine-readable companion, one row per actual Zindi upload: `submission_id, experiment_id, git_commit, model_version, feature_version, validation_version, tier1_score, tier2_mean, tier2_std, tier3_score, competition_score_local, public_lb_score, hypothesis, decision`. Only `C`-series rows are expected to compete for leaderboard position; `S`/`F`/`M`/`E` rows exist to generate evidence, and most of them are not expected to be the best score seen so far.

An explicit rule: **scientific evidence and robust CV outrank the public leaderboard**, not the reverse, unless there's specific evidence the public split represents the private one well — a real risk here given competitors are already at 40–80 submissions each, which is a strong signal some of them are chasing public-LB noise. A concrete go/no-go checkpoint: after the first Tier-2-validated `C`-series submission (targeted for week 3), compare local CV to the public LB score explicitly — a large or wrong-direction gap is the trigger to revisit Phase 2's assumptions immediately, not discover it in week 6 on top of several more models.

**Deliverables:** `docs/experiment_registry.md`; `submissions/leaderboard_log.csv`, seeded with its header row and the S/F/M/E/C schema from day one (template below); the go/no-go checkpoint result recorded explicitly once reached.

### Stage 17: Champion model gate

**What's done:** Before any candidate is called final, it passes: improves the competition score; the improvement holds across Tier 2 folds, not just on average; no leakage found (Stage 3/9 checks re-verified for this specific candidate); no unexplained train/test dependency (re-run adversarial validation on this candidate's residuals); reasonable, calibrated probability behavior; stable under geographic validation specifically; reproducible from a clean environment; and its `submission.csv` validates against `SampleSubmission.csv`'s schema exactly.

**Deliverables:** A short gate checklist per champion candidate in `docs/experiment_registry.md`.

---

*Phases 0–6 are the competition-critical work and the shared foundation. Phases 7–9 are substantively production/portfolio engineering — they start early (MLflow, tests, CI from Phase 0 onward) but the heaviest work lands once Phase 6 produces a locked champion, and continues past 19 Oct 2026 without pressure on the submission deadline.*

---

## Phase 7 — Production engineering

### Stage 18: Testing & CI/CD

**What's done:** `pytest` coverage for schemas (Stage 2), every feature transformer (Stages 6–9), the CV splitter (Stage 5), the leakage/OOF-encoding checks (Stage 9), and an end-to-end smoke test on a data sample verifying `submission.csv` matches `SampleSubmission.csv`'s schema exactly. Two GitHub Actions workflows: **CI** (lint, format check, tests, schema smoke test, pipeline smoke test — every push/PR) and **CD** (build Docker image → run tests against it → build artifact → deploy → health check — on release tag).

**Deliverables:** `tests/`, `.github/workflows/ci.yml`, `.github/workflows/release.yml`, a green CI badge in the README.

### Stage 19: Packaging & deployment (scope corrected from v1)

**What's done:** The trained, tagged champion pipeline served behind a **FastAPI** `/predict` endpoint (pydantic request validation mirroring the Stage 2 schema), with a **Streamlit** UI in front of it, both in one Docker image deployed to a single Hugging Face Space (Dockerfile-based Spaces run FastAPI+Streamlit together fine) — one host, not two, since splitting across providers adds CORS and cold-start risk for no portfolio benefit on a solo deadline project; the split-service architecture is documented in `docs/` as "how this decomposes at production scale," not actually built that way.

The interactive demo is honestly scoped rather than over-promised: since the model's strongest features are joined from `climate_features.csv` by record ID and can't be computed for an arbitrary user-typed new location/date, the UI lets a user **select an existing train/test record** (real location + date + real climate values) and perturb demographic fields (age, gender, zone) as a genuine what-if, plus a batch-CSV upload that reproduces the actual submission pipeline end to end. A live public-API climate lookup (so literally any lat/lon+date works) is a documented stretch goal for after the competition, not a Phase 7 requirement.

The UI carries an explicit disclaimer: *"This model estimates whether a mortality record is likely to belong to the climate-sensitive category defined by the competition dataset. It is a research/decision-support demonstration, not a clinical diagnostic or individual mortality-risk tool."*

**Deliverables:** `app/api/`, `app/streamlit_app/`, `Dockerfile`, one live public URL linked from the README.

### Stage 20: Lightweight monitoring (scoped down from v1's full drift-monitoring proposal)

**What's done:** A monitoring *module*, demonstrated rather than run as a 24/7 production service (there's no real-world label stream for a competition demo to validate against) — input-distribution checks (age/gender/zone/climate ranges vs. training distribution), prediction-distribution checks (mean predicted probability, positive rate), and data-quality checks (missingness, range/category violations, schema drift), runnable on demand or on a schedule against logged prediction requests.

**Deliverables:** `src/climate_health/monitoring/`; a short demonstration notebook/script and a paragraph in the README explaining this is an architecture demonstration, not a live SLA-backed service.

## Phase 8 — Responsible ML & documentation

### Stage 21: Interpretability

**What's done:** SHAP (global summary, dependence plots for `age` and the top climate/spatial features, individual-prediction explanations) alongside permutation importance, computed on the exact tagged `competition-submission-final` artifact — not a later-improved model, so the story told matches the score that was actually achieved.

**Deliverables:** `notebooks/05_interpretability.ipynb`; plots embedded in the model card and README.

### Stage 22: Model card & governance

**What's done:** `docs/model_card.md` — intended use, training data provenance and limitations (small dataset, single country, 15-year span, only 39 distinct training locations, stated plainly), performance by CV tier and by subgroup (age band, zone, spatial cluster), known failure modes (coordinates far from any training cluster). Every prediction traceable to model version, feature-pipeline version, training-data version, git SHA, and MLflow run — recorded as metadata alongside the champion artifact, not aspirational.

**Deliverables:** `docs/model_card.md`; the governance metadata attached to the served model artifact.

### Stage 23: README & architecture diagram

**What's done:** A top-level README telling the whole story in 90 seconds — problem, approach, key findings (age dominance, the confirmed geographic split driving the validation design, the calibration angle on the fixed threshold), results, a Mermaid architecture diagram, and links to the live demo, MLflow view, and CI badge.

**Deliverables:** `README.md`.

### Stage 24: Retrospective

**What's done:** Once the private leaderboard is revealed at close, compare the private score to the local Tier-2/Tier-3 CV estimates — the single most concrete piece of evidence for whether the validation architecture in Phase 2 actually worked, not just an assertion that it should have. Includes what would be tried with more time and the honestly-scoped statement of real-world relevance from §1.

**Deliverables:** `docs/retrospective.md`.

---

## Suggested timeline (7 weeks to close, revised for front-loaded competition work)

| Week | Focus | Phases/Stages |
|---|---|---|
| 1 (1–7 Sep) | Foundation, data forensics, deep EDA, first baselines | 0–1, Stage 10 |
| 2 | Validation architecture locked in, temporal/demographic features | 2, Stage 6 |
| 3 | Spatial + climate feature research; first Tier-2-validated submission; **go/no-go checkpoint vs. public LB** | Stages 7–9, Stage 16 checkpoint |
| 4 | Full model zoo, initial comparison table | Stages 11–12 |
| 5 | Hyperparameter tuning, ensembling | Stages 13–14 |
| 6 | Calibration, robustness, champion gate, final competition submissions | Stages 15, 17 |
| 7 (13–19 Oct) | Buffer for last CV-guided submissions before close; tag `competition-submission-final` | — |
| Post-close | Testing/CI-CD hardening, deployment, interpretability, docs, retrospective | Phases 7–8 |

MLflow, pytest, and pre-commit run from week 1 onward throughout — they are not a week-6 task, they're infrastructure that exists before the first experiment is logged.

## Tooling summary

| Concern | Tool |
|---|---|
| Environment/deps | Python 3.11, `pyproject.toml`, `uv` or `venv` |
| Data validation | `pandera` |
| Modeling | scikit-learn, LightGBM, XGBoost, CatBoost (+ HistGB/ExtraTrees/LogReg for ensemble diversity) |
| Spatial | k-means clustering on coordinates + climate normals |
| Tuning | Optuna |
| Tracking | MLflow (from Phase 4 onward) |
| Interpretability | SHAP |
| Testing | pytest |
| CI/CD | GitHub Actions (CI on push/PR, CD on tag) |
| Serving | FastAPI |
| Demo UI | Streamlit |
| Containerization | Docker (single image) |
| Hosting | Hugging Face Spaces (one Space, Dockerfile-based, API+UI together) |
| Diagramming | Mermaid (embedded in README) |

## Immediate next action

Stage 1 (repo scaffold) is still the right starting point — unchanged from v1, since everything above depends on it existing. What changes is what fills the scaffold next: Stage 3 (data forensics notebook) is now explicitly first, seeded directly from §0.1's already-verified findings, before general EDA — the reviews' strongest shared point was that this project needs to know how the data was put together before it starts modeling it, and that work has already started in this document.

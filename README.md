# Climate-Sensitive Mortality Prediction

Predicting whether a recorded death falls into a climate-sensitive category,
combining demographic, geographic, and climate/environmental data.
Competition + senior-level ML engineering portfolio project.

**Full project architecture, rationale, and stage-by-stage plan:**
see [`docs/PROJECT_BLUEPRINT.md`](docs/PROJECT_BLUEPRINT.md) — the single source
of truth for this repository.

Status: Phase 0 (Foundation) in progress. This README will be expanded per
Stage 23 once there is a trained model and live demo to describe.

## Quickstart

\`\`\`bash
conda env create -f environment.yml
conda activate climate-health
pip install -e ".[all]"
pytest tests/test_environment_smoke.py -v
\`\`\`

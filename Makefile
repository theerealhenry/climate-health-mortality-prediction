.PHONY: smoke-test lint format validate forensics eda train test serve

smoke-test:
	pytest tests/test_environment_smoke.py -v

lint:
	ruff check src tests
	black --check src tests

format:
	ruff check --fix src tests
	black src tests

validate:
	python -m climate_health.data.schemas

forensics:
	jupyter lab notebooks/00_data_forensics.ipynb

eda:
	jupyter lab notebooks/01_eda.ipynb

test:
	pytest tests/ -v

train:
	python -m climate_health.models.train

serve:
	uvicorn app.api.main:app --reload

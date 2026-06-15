.PHONY: install dev test lint format serve dashboard docker-up docker-down models clean

# === Setup ===
install:
	python3 -m venv .venv
	.venv/bin/pip install -e .

dev:
	python3 -m venv .venv
	.venv/bin/pip install -e ".[dev]"
	cp -n .env.example .env 2>/dev/null || true

# === Development ===
test:
	.venv/bin/pytest tests/ -v --tb=short

test-unit:
	.venv/bin/pytest tests/unit/ -v --tb=short

test-integration:
	.venv/bin/pytest tests/integration/ -v --tb=short

test-cov:
	.venv/bin/pytest tests/ -v --cov=src --cov-report=html --cov-report=term

lint:
	.venv/bin/ruff check src/ tests/

format:
	.venv/bin/ruff format src/ tests/

typecheck:
	.venv/bin/mypy src/

# === Run ===
serve:
	.venv/bin/uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload

dashboard:
	cd dashboard && npm run dev

# === Infrastructure ===
docker-up:
	docker compose up -d postgres redis

docker-down:
	docker compose down

# === Models ===
models:
	.venv/bin/python -m src.cli download-models

# === Data ===
data-list:
	.venv/bin/python scripts/download_data.py list-sources

data-trackid:
	.venv/bin/python scripts/download_data.py trackid3x3

data-sportvu:
	.venv/bin/python scripts/download_data.py sportvu

# === Database ===
db-migrate:
	.venv/bin/alembic upgrade head

db-revision:
	.venv/bin/alembic revision --autogenerate -m "$(msg)"

# === Cleanup ===
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf .coverage htmlcov/ dist/ build/ *.egg-info

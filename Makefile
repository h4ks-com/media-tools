.PHONY: install fix precommit format format-check lint lint-fix imports imports-check typecheck dead-code unused-deps security audit test coverage build quality ci docker-build run

install:
	uv sync
	uv run prek install

precommit: fix

fix: format imports lint-fix

format:
	uv run ruff format .

format-check:
	uv run ruff format --check .

lint:
	uv run ruff check .

lint-fix:
	uv run ruff check --fix .

imports:
	uv run ruff check --select I --fix .

imports-check:
	uv run ruff check --select I .

typecheck:
	uv run mypy

dead-code:
	uv run vulture src/media_tools tests

unused-deps:
	uv run deptry .

security:
	uv run bandit -c pyproject.toml -r src/media_tools

audit:
	uv run --with pip pip-audit

test:
	uv run pytest

coverage:
	uv run pytest --cov --cov-report=term-missing

build:
	uv build

quality: format-check lint typecheck imports-check dead-code unused-deps security audit coverage build
	@echo "quality gate passed"

ci: quality

docker-build:
	docker build -t media-tools .

run:
	uv run uvicorn media_tools.app:app --reload --port 8080

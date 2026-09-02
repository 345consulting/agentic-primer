# Copyright (c) 2026 345 Consulting, LLC.
# Proprietary and Confidential. All rights reserved.
#
# Run every recipe from THIS directory.

_default:
    @just --list

# every chapter on disk, in order, with its own one-line summary
list:
    PYTHONPATH=src uv run python -m support.chapters

# every chapter that has been run, on one page, in reading order
book:
    PYTHONPATH=src uv run python -m support.view

# run one chapter against the mock model, and open its trace
run CHAPTER:
    PYTHONPATH=src uv run python -m chapters.{{CHAPTER}}

# run one chapter against a real provider — requires DEEPSEEK_API_KEY
run-live CHAPTER:
    PYTHONPATH=src uv run python -m chapters.{{CHAPTER}} live

clean:
    rm -rf .pytest_cache .mypy_cache .ruff_cache out
    find . -name __pycache__ -type d -prune -exec rm -rf {} +

fmt:
    uv run ruff format src tests

# the quality gate: zero violations, zero errors, zero failures
gate:
    uv run ruff check src tests
    uv run ruff format --check src tests
    uv run mypy
    PYTHONPATH=src uv run basedpyright
    uv run pytest

lint:
    uv run ruff check src tests

test *ARGS:
    uv run pytest {{ARGS}}

type:
    uv run mypy
    PYTHONPATH=src uv run basedpyright

# Copyright (c) 2022-2026 345 Consulting, LLC
# Licensed under the MIT License. See LICENSE in the project root.
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

# run one chapter and write its page. Add `live` for the real model provider,
# which needs DEEPSEEK_API_KEY set and costs money:  just run ch01_... live
run CHAPTER *LIVE:
    PYTHONPATH=src uv run python -m run {{CHAPTER}} {{LIVE}}

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

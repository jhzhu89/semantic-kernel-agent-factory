# Makefile for semantic-kernel-agent-factory development

.PHONY: help install install-dev test lint format type-check build clean docs

help:  ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install:  ## Install package in production mode
	pip install .

install-dev:  ## Install package and development dependencies
	pip install -e .
	pip install -r dev-requirements.txt
	pre-commit install

test:  ## Run tests
	pytest

test-cov:  ## Run tests with coverage
	pytest --cov=agent_factory --cov-report=term-missing --cov-report=html

lint:  ## Run linting (ruff)
	ruff check agent_factory

format:  ## Format code (black + isort)
	black agent_factory
	isort agent_factory

type-check:  ## Run type checking (mypy)
	mypy agent_factory

build:  ## Build package
	python -m build

clean:  ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +

docs:  ## Build documentation
	sphinx-build -b html docs docs/_build

dev-setup:  ## Quick development setup
	./setup-dev.sh

all: format lint type-check test  ## Run all quality checks

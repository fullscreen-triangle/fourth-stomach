.PHONY: help install install-dev test test-cov lint format clean docs

help:
	@echo "Fourth Stomach - Development Commands"
	@echo ""
	@echo "Available targets:"
	@echo "  install       Install package and core dependencies"
	@echo "  install-dev   Install with development dependencies"
	@echo "  test          Run test suite"
	@echo "  test-cov      Run tests with coverage report"
	@echo "  lint          Run linters (ruff, mypy)"
	@echo "  format        Format code with black"
	@echo "  clean         Remove build artifacts and cache"
	@echo "  docs          Build documentation"
	@echo "  demo-ctn      Run circulation transaction network demo"
	@echo "  demo-shadow   Run shadow network demo"
	@echo "  demo-gcf      Run graph completion finance demo"
	@echo "  demo-rep      Run representation module demo"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

test:
	pytest tests/ -v

test-cov:
	pytest tests/ --cov=src --cov-report=html --cov-report=term

lint:
	ruff check src/ tests/
	mypy src/

format:
	black src/ tests/
	ruff check --fix src/ tests/

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache
	rm -rf .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

docs:
	cd docs && make html

demo-ctn:
	python src/ctn/demo_circulation.py

demo-shadow:
	python src/ctn/demo_shadow_network.py

demo-gcf:
	python src/ctn/demo_graph_completion_finance.py

demo-rep:
	python src/representation/demo_representations.py

# Research paper compilation
papers:
	@echo "Compiling research papers..."
	cd docs/publication && \
	for file in *.tex; do \
		echo "Compiling $$file..."; \
		pdflatex -interaction=nonstopmode "$$file" > /dev/null 2>&1; \
	done
	@echo "Papers compiled successfully"

# Validation framework
validation:
	@echo "Running validation framework experiments..."
	@echo "See docs/publication/validation-framework.tex for protocols"


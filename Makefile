# ============================================================
# DeepShield Audio — Makefile
# Convenience targets for common development tasks
# ============================================================

PYTHON   := .venv/bin/python
PIP      := .venv/bin/pip
PYTEST   := .venv/bin/pytest
STREAMLIT:= .venv/bin/streamlit

.PHONY: help install setup test test-fast lint app train-quick clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# ── Environment ──────────────────────────────────────────────────────────────

install:  ## Create venv (Python 3.11) & install all dependencies
	@echo "Creating virtual environment with Python 3.11..."
	/opt/homebrew/opt/python@3.11/bin/python3.11 -m venv .venv
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -r requirements.txt
	$(PIP) install -e .
	@echo "✅ Installation complete. Activate with: source .venv/bin/activate"

setup: install  ## Alias for install

# ── Testing ──────────────────────────────────────────────────────────────────

test:  ## Run full test suite
	$(PYTEST) tests/ -v --tb=short

test-fast:  ## Run tests, skip slow ones
	$(PYTEST) tests/ -v --tb=short -m "not slow"

test-coverage:  ## Run tests with coverage report
	$(PYTEST) tests/ --cov=src --cov-report=html --cov-report=term-missing
	@echo "Coverage report → htmlcov/index.html"

# ── Application ───────────────────────────────────────────────────────────────

app:  ## Launch the Streamlit application
	$(STREAMLIT) run app/streamlit_app.py

# ── Training ──────────────────────────────────────────────────────────────────

train-quick:  ## Quick pipeline test (5% of data)
	$(PYTHON) -m src.trainer --model custom_cnn --quick

train-all:  ## Train all three models (requires dataset)
	$(PYTHON) -m src.trainer --model all --epochs 50

eval:  ## Evaluate trained models on eval split
	$(PYTHON) -m src.evaluator --model all --split eval

# ── Notebooks ─────────────────────────────────────────────────────────────────

notebook:  ## Launch Jupyter notebook server
	.venv/bin/jupyter notebook notebooks/

# ── Cleaning ──────────────────────────────────────────────────────────────────

clean:  ## Remove Python caches
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	find . -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true

clean-all: clean  ## Remove venv + all generated files (DESTRUCTIVE)
	rm -rf .venv htmlcov .coverage
	@echo "⚠️  Virtual environment removed. Run 'make install' to recreate."

# Variables for common paths
PYTHON_DIRS = src/ tests/
DOCS_DIR = docs

.PHONY: install install_dev uninstall test clean lint format docs docs_serve docs_clean

install:
	pip install -e .

install_dev:
	pip install nanobind scikit-build-core[pyproject]
	pip install --no-build-isolation -ve .[docs]

uninstall:
	pip uninstall lotf

test:
	python -m pytest tests

lint:
	ruff check $(PYTHON_DIRS)
	ruff format --check $(PYTHON_DIRS)

format:
	ruff check --fix $(PYTHON_DIRS)
	ruff format $(PYTHON_DIRS)

clean:
	rm -rf build/
	rm -rf _skbuild/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} +
	$(MAKE) docs_clean

docs:
	cd $(DOCS_DIR) && sphinx-build -b html . _build/html

docs_serve:
	cd $(DOCS_DIR) && sphinx-autobuild . _build/html --host 0.0.0.0 --port 8000 --watch ../src

docs_clean:
	rm -rf $(DOCS_DIR)/_build/


.PHONY: run test test-fast lint format install install-dev clean

run:
	streamlit run app.py

test:
	python3 -m pytest tests/ -v

test-fast:
	python3 -m pytest tests/ -v -m "not slow"

lint:
	python3 -m ruff check .

format:
	python3 -m ruff format .

install:
	pip3 install -r requirements.txt

install-dev:
	pip3 install -r requirements-dev.txt

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .ruff_cache -exec rm -rf {} +
	rm -rf htmlcov .coverage

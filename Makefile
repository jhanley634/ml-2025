
SHELL := bash
ACTIVATE := source .venv/bin/activate

all:
	ls -l

.venv:
	which uv || curl -LsSf https://astral.sh/uv/install.sh | sh
	uv venv --python=python3.13

install: .venv
	sort -o requirements.txt{,}
	$(ACTIVATE) && uv pip compile --quiet requirements.txt -o requirements.lock
	$(ACTIVATE) && uv pip install --upgrade -r requirements.lock
	$(ACTIVATE) && pre-commit install

STRICT = --strict --warn-unreachable --ignore-missing-imports

ruff-check:
	$(ACTIVATE) && ruff check --fix && black .
lint: ruff-check
	$(ACTIVATE) && pyright .
	$(ACTIVATE) && mypy $(STRICT) .

CACHES := .mypy_cache/ .ruff_cache/

clean-caches:
	rm -rf $(CACHES)
clean: clean-caches
	rm -rf .venv/

.PHONY: all install ruff-check lint clean-caches clean


ACTIVATE := source .venv/bin/activate

all:
	ls -l

.venv:
	which uv || curl -LsSf https://astral.sh/uv/install.sh | sh
	uv venv --python=python3.13

install: .venv
	$(ACTIVATE) && uv pip compile --quiet requirements.txt -o requirements.lock
	$(ACTIVATE) && uv pip install -r requirements.lock
	$(ACTIVATE) && pre-commit install

STRICT = --strict --warn-unreachable --ignore-missing-imports

ruff-check:
	$(ACTIVATE) && ruff check --fix
lint: ruff-check
	$(ACTIVATE) && pyright .
	$(ACTIVATE) && mypy $(STRICT) .

clean:
	rm -rf .venv

.PHONY: clean

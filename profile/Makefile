
SHELL := bash
ACTIVATE := source .venv/bin/activate

CSV ?= /tmp/data.csv

profile: .venv
	$(ACTIVATE) && ydata_profiling --silent $(CSV)

install: .venv
	sort -o requirements.txt{,}
	$(ACTIVATE) && uv pip compile --quiet requirements.txt -o requirements.lock
	$(ACTIVATE) && uv pip install -r requirements.lock

.venv:
	which uv || curl -LsSf https://astral.sh/uv/install.sh | sh
	uv venv --python=python3.12

clean:
	rm -rf .venv/

.PHONY: install clean


SHELL := bash
ACTIVATE := source .venv/bin/activate
PYTHONPATH := .:../..
ENV := env PYTHONPATH=$(PYTHONPATH)

all: toc
toc:
	# table of contents, a list of targets
	awk '/^[a-z]*:/ {print $$1}' Makefile

.venv:
	uv venv --python=python3.9

install: .venv
	sort -o requirements.txt{,}
	$(ACTIVATE) && uv pip install -r requirements.txt

test:
	$(ACTIVATE) && $(ENV) python -c 'import irvine.air_quality.aq_etl'
	$(ACTIVATE) && $(ENV) ./as_demo.py

lint:
	$(ACTIVATE) && ruff check .

clean:
	rm -rf .venv/

.PHONY: all .venv install test lint clean

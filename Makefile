PIP := $(shell which pip)
PYTHON := $(shell which python)

install: requirements.txt
	$(PIP) install -r requirements.txt

download_data:
	./sh-scripts/download_data.sh

download_spacy_model:
	$(PYTHON) -m spacy download en_core_web_lg

test:
	$(PYTHON) -m unittest

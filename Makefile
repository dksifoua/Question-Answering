.PHONY: tests

PIP := $(shell which pip)
PYTHON := $(shell which python)

install: requirements.txt
	$(PIP) install -r requirements.txt

download_data:
	./sh-scripts/download_data.sh

download_spacy_model:
	$(PYTHON) -m spacy download en_core_web_lg

drqa_process_data:
	$(PYTHON) -m py_scripts.drqa.process_data

drqa_build_vocabulary:
	$(PYTHON) -m py_scripts.drqa.build_vocabulary

tests:
	$(PYTHON) -m unittest

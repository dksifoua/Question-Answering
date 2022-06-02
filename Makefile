download_data:
	@./sh-scripts/download_data.sh

setup_spacy_model:
	@python -m spacy download en_core_web_lg
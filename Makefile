.PHONY: data run test lint
CATEGORY ?= Health_and_Personal_Care
BASE = https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw

data:
	mkdir -p data/raw
	curl -fL -o data/raw/$(CATEGORY).jsonl.gz $(BASE)/review_categories/$(CATEGORY).jsonl.gz
	curl -fL -o data/raw/meta_$(CATEGORY).jsonl.gz $(BASE)/meta_categories/meta_$(CATEGORY).jsonl.gz

run:
	python -m reviewpipe all --config config.toml

test:
	pytest --cov=reviewpipe --cov-branch

lint:
	ruff check . && ruff format --check .

# One-command pipeline for the WSB direction project.
#
#   make data      # corpus -> labels -> tickers -> price labels -> dataset
#   make baselines # B0-B6
#   make train     # primary seed (42)
#   make eval      # primary run + bootstrap
#   make serve     # HTTP API on :8080
#   make test      # fast tests (no transformer load)
#
# Override variables on the command line, e.g.
#   make train SEED=43 RUN_ID=finbert_h5_seed43
#   make eval RUN_ID=finbert_h5_seed43 BOOTSTRAP=1000

# Use the project virtualenv when present, otherwise the active python.
PYTHON ?= $(if $(wildcard ../.venv/bin/python),../.venv/bin/python,python)
SEED ?= 42
RUN_ID ?= finbert_h5_seed$(SEED)
BOOTSTRAP ?= 1000
PUBLISH ?=
PORT ?= 8080
HOST ?= 0.0.0.0

.DEFAULT_GOAL := help
.PHONY: help data baselines train train-all experiments eval aggregate walk-forward gold-audit ticker-gold-sample ticker-gold-audit serve test test-all lint clean-lite

help:
	@echo "Targets:"
	@echo "  data        build the 16k corpus, labels, ticker links and direction dataset"
	@echo "  baselines   run B0-B6 and write reports/baselines*"
	@echo "  train       train the primary model (SEED=$(SEED), RUN_ID=$(RUN_ID))"
	@echo "  train-all   train seeds 42/43/44"
	@echo "  experiments run seeds 43/44 plus the pre-registered ablation matrix"
	@echo "  eval        evaluate RUN_ID=$(RUN_ID) (PUBLISH=1 copies root artifacts)"
	@echo "  aggregate   summarise seeds and ablations"
	@echo "  walk-forward run folds 2023/2024/2025 (train <= Y-2, val = Y-1, test = Y)"
	@echo "  gold-audit  score the 500-post human class-label audit (kappa)"
	@echo "  ticker-gold-sample  create the 300-post ticker gold sample"
	@echo "  ticker-gold-audit   score ticker precision/recall on that sample"
	@echo "  serve       start the HTTP API on $(HOST):$(PORT)"
	@echo "  test        fast pytest suite; test-all includes the real-model smoke test"
	@echo "  lint        byte-compile src/ and tests/"

data:
	$(PYTHON) src/collect_posts.py --target-total 16000 --per-year 2000 --seed 42
	$(PYTHON) src/label_posts.py
	$(PYTHON) src/audit_classifier.py
	$(PYTHON) src/extract_tickers.py
	$(PYTHON) src/build_dataset.py --all-horizons
	$(PYTHON) src/embeddings.py

baselines: data
	$(PYTHON) src/baselines.py --config configs/eval.yaml

train:
	$(PYTHON) src/train.py --config configs/model.yaml --seed $(SEED) --run-id $(RUN_ID)

train-all:
	$(PYTHON) src/run_experiments.py --preset core --only finbert_h5_seed --bootstrap $(BOOTSTRAP)

experiments:
	$(PYTHON) src/run_experiments.py --preset core --bootstrap $(BOOTSTRAP)

eval:
	$(PYTHON) src/evaluate.py --run-id $(RUN_ID) --bootstrap $(BOOTSTRAP) $(if $(PUBLISH),--publish,)

aggregate:
	$(PYTHON) src/aggregate_results.py

walk-forward:
	$(PYTHON) src/walk_forward.py --folds 2023,2024,2025 --bootstrap $(BOOTSTRAP)

gold-audit:
	$(PYTHON) src/score_gold_audit.py

ticker-gold-sample:
	$(PYTHON) src/sample_ticker_gold.py

ticker-gold-audit:
	$(PYTHON) src/score_ticker_gold.py

serve:
	$(PYTHON) src/serve.py --run-id $(RUN_ID) --host $(HOST) --port $(PORT)

test:
	$(PYTHON) -m pytest tests -q -m "not slow"

test-all:
	WSB_TEST_REAL_MODEL=1 $(PYTHON) -m pytest tests -q

lint:
	$(PYTHON) -m compileall -q src tests

clean-lite:
	rm -rf __pycache__ src/__pycache__ tests/__pycache__ .pytest_cache

# Mon Amo Cost — Makefile de reproduction one-shot
# Pour le bonus reproductibilité du challenge EvalLLM 2026

.PHONY: help up down logs build ingest reproduce eval clean

help:
	@echo "Commandes disponibles :"
	@echo "  make build      — construit l'image Docker du backend (Tesseract+Poppler inclus)"
	@echo "  make up         — démarre Postgres + backend en arrière-plan"
	@echo "  make down       — stoppe et supprime les conteneurs (volume Postgres conservé)"
	@echo "  make logs       — suit les logs du backend"
	@echo "  make ingest     — ingère le corpus PDF (Experimental/DATA/Corpus_raw/)"
	@echo "  make eval       — lance run_challenge + métriques RI/génération sur le sample"
	@echo "  make reproduce  — séquence complète : build + up + ingest + eval"
	@echo "  make clean      — supprime conteneurs ET volume Postgres (reset complet)"

build:
	docker compose build

up:
	docker compose up -d
	@echo ""
	@echo "Postgres + backend démarrés. Vérification :"
	@docker compose ps

down:
	docker compose down

logs:
	docker compose logs -f backend

ingest:
	@echo "Ingestion du corpus depuis Experimental/DATA/Corpus_raw/ ..."
	docker compose exec backend uv run python ingest.py

eval:
	@echo "Run challenge sur sample_queries.json + évaluations ..."
	docker compose exec backend uv run python run_challenge.py \
		--input ../Experimental/DATA/training/sample_queries.json \
		--out-task1 ../Experimental/DATA/runs/repro_task1.json \
		--out-task2 ../Experimental/DATA/runs/repro_task2.json
	docker compose exec backend uv run python eval_retrieval.py \
		--predictions ../Experimental/DATA/runs/repro_task1.json \
		--gold ../Experimental/DATA/training/sample_queries.json \
		--output ../Experimental/DATA/runs/repro_eval_retrieval.json

reproduce: build up
	@echo ""
	@echo "Attente que le backend soit healthy (max 60s)..."
	@sleep 10
	@$(MAKE) ingest
	@$(MAKE) eval
	@echo ""
	@echo "Reproduction terminée. Résultats dans Experimental/DATA/runs/repro_*.json"
	@echo "Métriques RI :"
	@docker compose exec backend cat ../Experimental/DATA/runs/repro_eval_retrieval.json 2>/dev/null | head -30

clean:
	docker compose down -v
	@echo "Volume Postgres + conteneurs supprimés."

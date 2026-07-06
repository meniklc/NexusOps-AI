# Convenience targets. `make help` lists them.
.DEFAULT_GOAL := help
.PHONY: help up down logs ingest eval test compile fmt

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

up: ## Build & start the full stack (db + backend + frontend)
	docker compose up --build

down: ## Stop the stack (add ARGS=-v to wipe the db volume)
	docker compose down $(ARGS)

logs: ## Tail backend logs
	docker compose logs -f backend

ingest: ## Ingest the sample docs into pgvector
	docker compose exec backend python -m scripts.ingest sample_docs

eval: ## Run the evaluation harness
	docker compose exec backend python -m app.eval.run_eval

test: ## Run the offline unit tests (no services required)
	pytest -q

compile: ## Byte-compile all sources (fast syntax check)
	python -m compileall -q app scripts frontend tests

PACKAGES := packages/gridtools packages/chirpdetector

.PHONY: sync check lint typecheck test

sync:
	uv sync

lint:
	uv run ruff check . $(PACKAGES)
	uv run ruff format --check . $(PACKAGES)

typecheck:
	@for p in $(PACKAGES); do uv run pyright -p $$p || exit 1; done

test:
	uv run pytest -q $(PACKAGES)

check: lint typecheck test

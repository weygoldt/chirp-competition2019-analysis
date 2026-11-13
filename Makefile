PACKAGES := packages/gridtools packages/chirpdetector

.PHONY: sync check lint typecheck test paper-test

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

# The manuscript analysis (paper/, own uv project): loader tests only.
paper-test:
	cd paper && unset VIRTUAL_ENV && uv run pytest -q

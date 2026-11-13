PACKAGES := packages/gridtools packages/chirpdetector

.PHONY: sync check lint typecheck test docs

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

# Documentation sites of both packages (MkDocs Material, strict: broken links fail)
MKDOCS := uvx --from mkdocs==1.6.1 --with mkdocs-material==9.8.1 mkdocs

docs:
	@for p in $(PACKAGES); do (cd $$p && NO_MKDOCS_2_WARNING=1 $(MKDOCS) build --strict) || exit 1; done

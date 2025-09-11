# Contributing to dspy-code

Thanks for your interest! Contributions are welcome — issues, docs, and code.

## Setup

- Install Python 3.10+
- Install uv: `pip install uv`
- Sync deps: `uv sync`
- Run CLI help: `uv run dspy-code --help`

## Dev Workflow

- Run unit tests: `uv run python -m unittest discover -s tests -v`
- Lint (optional): add your favorite linter (ruff, flake8) — not enforced yet.
- Validate lightweight stack:
  - `dspy-code lightweight_init --workspace $(pwd) --logs ./logs --db auto`
  - `docker compose -f docker/lightweight/docker-compose.yml build`
  - `docker compose -f docker/lightweight/docker-compose.yml up -d`
  - `docker compose -f docker/lightweight/docker-compose.yml logs -f dspy-agent`

## Pull Requests

- Keep changes focused; update docs for any user-facing changes.
- If adding new modules, include minimal unit tests where practical.
- Match the existing code style; avoid unrelated refactors.

## Code of Conduct

- Be respectful and constructive.
- No harassment or discrimination.

## License

- This project is licensed under Apache-2.0. By contributing, you agree that your contributions will be licensed under the same terms.

## Links

- DSPy: https://github.com/stanfordnlp/dspy
- RedDB (open): https://github.com/redbco/redb-open


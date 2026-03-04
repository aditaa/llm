# Repository Guidelines

## Project Structure & Module Organization
The codebase is a Python LLM-from-scratch scaffold.
- `src/llm/`: core package (`tokenizer.py`, `cli.py`, `model.py`)
- `tests/`: unit tests (currently tokenizer coverage)
- `docs/`: architecture and roadmap docs
- `information/`: references, imported notes, and source material
- `artifacts/`: generated outputs such as vocab/checkpoints (gitignored)

Keep modules single-purpose and expand by domain (for example: `src/llm/training.py`, `src/llm/data.py`).

## Build, Test, and Development Commands
Use the `Makefile` as the source of truth:
- `make test`: run `unittest` test suite
- `make lint`: run Ruff lint checks
- `make format`: run Black formatter
- `make typecheck`: run MyPy on `src/`
- `make smoke`: run a minimal CLI smoke test

Initial setup:
`python3 -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]"`

## Coding Style & Naming Conventions
- Python 3.10+, 4-space indentation, UTF-8 files
- Max line length: 100 (Black/Ruff configured in `pyproject.toml`)
- Use `snake_case` for functions/modules, `PascalCase` for classes, `UPPER_CASE` for constants
- Add type hints for public functions; MyPy is configured with `disallow_untyped_defs = true`

## Testing Guidelines
- Framework: `unittest` (`tests/test_*.py`)
- Mirror source layout when adding tests (example: `src/llm/data.py` -> `tests/test_data.py`)
- Cover edge cases and failure modes, not only happy paths
- Run `make test` locally before each commit

## Commit & Pull Request Guidelines
Follow concise, imperative commit subjects (<= 72 chars), for example:
- `Add tokenizer save/load round-trip tests`
- `Implement corpus stats CLI command`

PRs should include:
- What changed and why
- How it was validated (`make test`, `make lint`, sample output if relevant)
- Linked issue/ticket when applicable

Keep PR scope narrow; split refactors and features into separate PRs.

## Security & Configuration Tips
- Never commit secrets or credentials
- Keep generated files in `artifacts/` and out of git history
- Prefer environment variables for machine-specific settings

## Reference Material Workflow
- Store reusable project references in `information/`
- Start with `information/README.md` for curated external links
- Keep `information/raschka-reference-notes.md` updated when Raschka source material informs implementation
- When adding a new source, include a short note on why it matters to this codebase

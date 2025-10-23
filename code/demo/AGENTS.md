# Repository Guidelines

## Project Structure & Module Organization
- `code/`: main source code.
  - `demo.py`: end-to-end data prep, training, evaluation, and plotting.
  - `persistent_multiscale.py`: train-once/save and reuse `MultiScaleModel` per zone.
  - `window_map.py`: map generation utilities.
- Data assets: `data.parquet`, `taxi-zone-lookup.csv`, `edge_weight_matrix_with_flow.csv` in project root or `code/`.
- Outputs: CSVs and checkpoints (e.g., `checkpoints_multiscale/`).

## Build, Test, and Development Commands
- Run demo end-to-end:
  - `python code/demo.py`
- Train-once + predict (no retrain when re-used):
  - `python code/persistent_multiscale.py --data data.parquet --target "2021-03-05 12:00"`
- Generate merged results CSVs and plots: produced by `demo.py` run.

## Coding Style & Naming Conventions
- Python 3.9+; prefer type hints in new code.
- Indentation: 4 spaces; max line length ~100 chars.
- Naming: modules `snake_case.py`, classes `CamelCase`, functions/vars `snake_case`.
- Keep imports standard → third-party → local; avoid unused imports.
- Do not add license headers unless requested.

## Testing Guidelines
- Prefer small, deterministic functions; add unit tests to `tests/` (create if absent) using `pytest`.
- Name tests `test_*.py`; run with `pytest -q`.
- For data-driven code, add lightweight fixtures and seed RNGs where applicable.

## Commit & Pull Request Guidelines
- Commits: imperative mood, concise scope-first subject (e.g., "Add train-once manager for MultiScaleModel").
- Include rationale in body when changing behavior or data formats.
- PRs: clear description, linked issues, reproduction steps, before/after notes, and sample commands.
- Touch only related files; keep changes minimal and consistent with existing style.

## Security & Configuration Tips
- Do not commit large data or secrets; use environment variables or `.env` (gitignored).
- Checkpoints and generated CSVs should go under `checkpoints_multiscale/` or a dedicated `outputs/` folder.

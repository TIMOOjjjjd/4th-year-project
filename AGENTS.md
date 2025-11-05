# Repository Guidelines

## Project Structure & Module Organization
- Primary pipeline scripts (`demo.py`, `persistent_multiscale.py`, `V2_manager.py`, `window_map.py`) live in `code/demo/code` alongside generated artifacts (HTML map, CSV metrics, checkpoints). Keep `data.parquet`, lookup tables, and GeoJSON assets in this directory so relative paths stay intact.
- Use `code/test` for quick experiments (`lstm.py`, `lstm1.py`); promote stable utilities into `code/demo/code` and tidy imports before committing.
- Large binaries (presentations, PDFs) already sit with the demo; avoid duplicating them elsewhere and call out shared locations in pull requests.

## Build, Test, and Development Commands
- Create a virtual environment before development: `python -m venv .venv && source .venv/bin/activate`.
- Install dependencies once per environment: `pip install torch torch_geometric pandas numpy scikit-learn matplotlib pyarrow folium branca PyQt5 PyQtWebEngine`.
- Refresh forecasts with `python code/demo/code/demo.py`; it retrains on `data.parquet` and updates `nyc_taxi_prediction_map.html`.
- Validate visualization and interaction via `python code/demo/code/window_map.py`.
- Perform incremental or zone-specific retraining using `python code/demo/code/V2_manager.py --data code/demo/code/data.parquet --checkpoints checkpoints_v2 --zones 1 2 3`.

## Coding Style & Naming Conventions
- Follow PEP 8 with four-space indentation and ≤100-character lines to mirror existing modules.
- Use snake_case for modules/functions, PascalCase for classes, and concise docstrings around data loaders, feature engineering, and training loops.
- Prefer `pathlib.Path` for filesystem access and guard CUDA branches with `torch.cuda.is_available()` to keep CPU-only workflows stable.

## Testing Guidelines
- Maintain lightweight smoke tests under `code/test/`; name additions `test_<feature>.py` and print key metrics (loss, MAE, RMSE).
- When logic is reusable, wrap it in pytest functions and run `pytest code/test` before opening a PR.
- Capture evaluation results in CSV logs (e.g. append to `hourly_metrics.csv`) so reviewers can compare runs without recomputing everything.

## Commit & Pull Request Guidelines
- Adopt the `<type>: <imperative summary>` format seen in history (`fix: stabilize mae update`, `add: rolling trainer`) and keep commits focused.
- In PR descriptions, list commands executed, summarize metric deltas, and attach new heatmap or GUI screenshots when visuals change.
- Reference related issues, call out required data downloads from `README.md`, and document new dependencies or configuration switches for reviewers.

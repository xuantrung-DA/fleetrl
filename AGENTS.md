# Repository Guidelines

## Project Structure & Module Organization

FleetRL is a Python 3.12 warehouse-fleet dispatch project using a `src/` layout.

- `src/fleetrl/` contains the package: simulator, environment, planner, RL
  utilities, CLI, and metrics.
- `src/fleetrl/methods/` holds one implementation per dispatch method; register
  new methods through `methods/registry.py`.
- `src/fleetrl/optimization/` contains MILP/OR-Tools logic.
- `tests/` mirrors the package by concern (`test_planner.py`, `test_rl.py`, etc.).
- `configs/` contains reproducible YAML settings; method-specific files belong in
  `configs/methods/` and controls in `configs/controls/`.
- `scripts/` provides audit, training, and checksum helpers. `docs/` and
  `reports/` hold guidance and verification evidence; place generated runtime
  outputs under ignored `runs/`.

## Build, Test, and Development Commands

Run all commands from the repository root with the isolated virtual environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"  # install package and tools
.\.venv\Scripts\python.exe -m fleetrl doctor            # validate runtime
.\.venv\Scripts\python.exe -m pytest -q                 # run test suite
.\.venv\Scripts\python.exe -m ruff check src tests scripts
.\.venv\Scripts\python.exe -m ruff format --check src tests scripts
.\.venv\Scripts\python.exe scripts/update_checksums.py --check
```

Use `ruff format src tests scripts` to apply formatting. Run a targeted test,
for example `-m pytest tests/test_planner.py -q`, while iterating.

## Coding Style & Naming Conventions

Use four-space indentation, type hints, and Python 3.12 features compatible with
the declared runtime. Ruff enforces a 100-character line limit plus import and
basic error checks. Use `snake_case` for modules, functions, variables, and YAML
keys; `PascalCase` for classes; and descriptive method files such as
`ppo_cpsat.py`. Keep simulation, optimization, and reporting responsibilities
separate rather than adding cross-module shortcuts.

## Testing Guidelines

Add or update focused `tests/test_<area>.py` coverage for behavior changes.
Mark full-engine/training tests with `@pytest.mark.integration` and extended
acceptance cases with `@pytest.mark.slow`. Do not treat historical files in
`reports/` as proof for modified source: rerun relevant checks into a new
`runs/` directory.

## Commit & Pull Request Guidelines

The available history contains only `Initial commit`, so no mature project
convention exists yet. Use short imperative conventional-style messages, e.g.
`feat(planner): add charging constraint` or `fix(metrics): handle empty runs`.
Before committing, run `git diff --check`, lint, relevant tests, and regenerate
`SHA256SUMS.txt` after repository-file changes. PRs should state the scenario
affected, config used, validation commands/results, and link related issues;
include screenshots only for UI or visual-output changes.

# Development and Git preparation

Run commands from the repository root using Python 3.12. Install the validated
runtime dependencies following [READY_TO_TRAIN.md](READY_TO_TRAIN.md), then install
the development tools:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m ruff check src tests scripts
.\.venv\Scripts\python.exe -m ruff format --check src tests scripts
.\.venv\Scripts\python.exe -m pytest -q
```

To apply formatting:

```powershell
.\.venv\Scripts\python.exe -m ruff format src tests scripts
```

The repository includes source, tests, configs, scripts, documentation and the
existing verification reports. Git ignores virtual environments, Python caches,
local environment files, build output, `runs/` and `.baseline/`. The historical
`.baseline/fleetrl-v1.zip` archive is optional and remains a local artifact.

Reports describe the source hashes recorded when they were produced. Formatting
and import cleanup change the source hash even when behavior is preserved. Old
reports remain historical evidence; rerun acceptance into a new directory to
verify the current source. Resume and study identity checks can reject artifacts
from before cleanup because the source hash changed.

```powershell
.\.venv\Scripts\python.exe -m fleetrl ready-check --output runs/ready_after_cleanup
```

`SHA256SUMS.txt` identifies the current files prepared for Git, excluding itself
and ignored local artifacts. Regenerate it after changing repository files.

```powershell
.\.venv\Scripts\python.exe scripts/update_checksums.py
.\.venv\Scripts\python.exe scripts/update_checksums.py --check
```

The manifest hashes working-tree bytes. Text files use LF, PowerShell scripts use
CRLF, and historical reports retain their original bytes through `.gitattributes`.

Before committing, inspect the files and diff:

```powershell
git status --short
git diff --check
git diff
```

Configure a remote and push only after selecting the destination repository.

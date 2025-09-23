#!/usr/bin/env bash
set -euo pipefail

if ! command -v python >/dev/null 2>&1; then
  echo "python not found; please install Python 3 and retry" >&2
  exit 127
fi

if ! python - <<'PY'
import sys
try:
    import pytest  # noqa: F401
except Exception:
    sys.exit(1)
else:
    sys.exit(0)
PY
then
  echo "pytest not found; installing locally (user)" >&2
  python -m pip install --user pytest >/dev/null
fi

echo "Running tests..."
python -m pytest -q

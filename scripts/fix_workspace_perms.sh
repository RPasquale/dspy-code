#!/usr/bin/env bash
set -euo pipefail

# Ensure a host directory is writable by container user 10001:10001.
# Usage: ./scripts/fix_workspace_perms.sh /absolute/path

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 /absolute/path" >&2
  exit 2
fi

TARGET="$1"
if [[ ! -d "$TARGET" ]]; then
  echo "Directory does not exist: $TARGET" >&2
  exit 2
fi

echo "[perms] Setting group ownership to GID 10001 and enabling group rwX..."
echo "[perms] You may need sudo for the chgrp step."

if command -v sudo >/dev/null 2>&1; then
  sudo chgrp -R 10001 "$TARGET" || true
else
  chgrp -R 10001 "$TARGET" || true
fi

chmod -R g+rwX "$TARGET"

echo "[perms] Done. If issues persist, consider also:"
echo "  sudo chown -R :10001 $TARGET"
echo "  find $TARGET -type d -exec chmod g+s {} +  # keep group on new files"


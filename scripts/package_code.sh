#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${1:-$REPO_ROOT/artifacts/release}"
ARCHIVE_NAME="lsrna_reproduction_code_v1.zip"
OUTPUT_DIR="$(mkdir -p "$OUTPUT_DIR" && cd "$OUTPUT_DIR" && pwd)"
ARCHIVE_PATH="$OUTPUT_DIR/$ARCHIVE_NAME"

if [[ -e "$ARCHIVE_PATH" ]]; then
    echo "Refusing to overwrite existing archive: $ARCHIVE_PATH" >&2
    exit 1
fi

cd "$REPO_ROOT"
git ls-files --cached --others --exclude-standard \
    | grep -Ev '^(lsr_training/save/|test/)' \
    | zip -1 -q "$ARCHIVE_PATH" -@

unzip -tq "$ARCHIVE_PATH"
(
    cd "$OUTPUT_DIR"
    sha256sum "$ARCHIVE_NAME" > "$ARCHIVE_NAME.sha256"
)
echo "Created: $ARCHIVE_PATH"

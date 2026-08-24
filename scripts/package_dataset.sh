#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $0 --data-root PATH [--output-dir PATH] [--bundle train|eval|rgb|smoke|full]"
}

DATA_ROOT=""
OUTPUT_DIR="artifacts/datasets"
BUNDLE="train"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --data-root)
            DATA_ROOT="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --bundle)
            BUNDLE="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ -z "$DATA_ROOT" ]]; then
    echo "--data-root is required." >&2
    exit 2
fi
if [[ ! -d "$DATA_ROOT" ]]; then
    echo "Dataset directory not found: $DATA_ROOT" >&2
    exit 1
fi
if ! command -v zip >/dev/null 2>&1; then
    echo "The 'zip' command is required." >&2
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="$(cd "$DATA_ROOT" && pwd)"
DATA_PARENT="$(dirname "$DATA_ROOT")"
DATA_NAME="$(basename "$DATA_ROOT")"
OUTPUT_DIR="$(mkdir -p "$OUTPUT_DIR" && cd "$OUTPUT_DIR" && pwd)"

case "$BUNDLE" in
    train)
        ARCHIVE_NAME="lsrna_train_latents_v1.zip"
        PATHS=(
            "$DATA_NAME/HR_sdxl_latent"
            "$DATA_NAME/LR_sdxl_latent"
            "$DATA_NAME/valid/HR"
            "$DATA_NAME/valid/HR_resized"
        )
        ;;
    eval)
        ARCHIVE_NAME="lsrna_eval_data_v1.zip"
        PATHS=(
            "$DATA_NAME/test"
            "$DATA_NAME/valid"
            "$DATA_NAME/captions.json"
        )
        ;;
    rgb)
        ARCHIVE_NAME="lsrna_rgb_intermediates_v1.zip"
        PATHS=("$DATA_NAME/HR" "$DATA_NAME/LR")
        ;;
    smoke)
        ARCHIVE_NAME="lsrna_smoke_data_v1.zip"
        PATHS=(
            "$DATA_NAME/HR_sdxl_latent"
            "$DATA_NAME/LR_sdxl_latent"
            "$DATA_NAME/valid/HR"
            "$DATA_NAME/valid/HR_resized"
            "$DATA_NAME/smoke_manifest.json"
        )
        ;;
    full)
        ARCHIVE_NAME="lsrna_openimages_full_v1.zip"
        PATHS=("$DATA_NAME")
        ;;
    *)
        echo "Unsupported bundle: $BUNDLE" >&2
        exit 2
        ;;
esac

ARCHIVE_PATH="$OUTPUT_DIR/$ARCHIVE_NAME"
REPORT_PATH="$OUTPUT_DIR/${ARCHIVE_NAME%.zip}.dataset-info.json"
MANIFEST_PATH="$OUTPUT_DIR/${ARCHIVE_NAME%.zip}.manifest.tsv"

if [[ -e "$ARCHIVE_PATH" ]]; then
    echo "Refusing to overwrite existing archive: $ARCHIVE_PATH" >&2
    exit 1
fi

python3 "$REPO_ROOT/scripts/check_dataset.py" \
    --data-root "$DATA_ROOT" \
    --samples 16 \
    --report "$REPORT_PATH"

(
    cd "$DATA_PARENT"
    find "${PATHS[@]}" -type f -printf '%p\t%s\n' | LC_ALL=C sort
) > "$MANIFEST_PATH"

echo "Creating $ARCHIVE_PATH"
echo "This can take a long time for the training bundle."
(
    cd "$DATA_PARENT"
    zip -1 -q -r "$ARCHIVE_PATH" "${PATHS[@]}"
)

echo "Testing ZIP integrity..."
unzip -tq "$ARCHIVE_PATH"
(
    cd "$OUTPUT_DIR"
    sha256sum "$ARCHIVE_NAME" > "$ARCHIVE_NAME.sha256"
)
echo "Created: $ARCHIVE_PATH"
echo "Checksum: $ARCHIVE_PATH.sha256"

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

INPUT_JSON="${1:-${SCRIPT_DIR}/sample_questions.json}"
OUTPUT_DIR="${2:-${SCRIPT_DIR}/out_csv}"
OUTPUT_PREFIX="${3:-langdist}"
REFERENCE_LANG="${4:-zh}"

ALPHA="${ALPHA:-0.4}"
BETA="${BETA:-0.2}"
GAMMA="${GAMMA:-0.3}"
LAM="${LAM:-0.1}"
DIM="${DIM:-256}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  else
    echo "Error: python3/python not found in PATH" >&2
    exit 1
  fi
fi

if [[ ! -f "${INPUT_JSON}" ]]; then
  echo "Error: input JSON not found: ${INPUT_JSON}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/calculate_distance.py" \
  --input-json "${INPUT_JSON}" \
  --output-dir "${OUTPUT_DIR}" \
  --output-prefix "${OUTPUT_PREFIX}" \
  --reference-lang "${REFERENCE_LANG}" \
  --alpha "${ALPHA}" \
  --beta "${BETA}" \
  --gamma "${GAMMA}" \
  --lam "${LAM}" \
  --dim "${DIM}"

echo "Done. CSV files are in: ${OUTPUT_DIR}"

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
DATASET_NAME=""
PASSTHROUGH=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset-name)
      if [[ $# -lt 2 ]]; then
        echo "--dataset-name requires a value" >&2
        exit 2
      fi
      DATASET_NAME="$2"
      shift 2
      ;;
    -h|--help)
      cat <<'EOF'
Usage: bash src/vae_preprocessing/latent_save.sh --dataset-name NAME [options]

Dataset names:
  fashionpedia
  people_clothing_segmentation (aliases: pcs, people-clothing-segmentation)

Options after --dataset-name are passed to the selected Python cache builder.
Common options include --batch, --chunk, --workers, --seed, --output, and --force.
EOF
      exit 0
      ;;
    *)
      PASSTHROUGH+=("$1")
      shift
      ;;
  esac
done

cd "${PROJECT_ROOT}"
case "${DATASET_NAME}" in
  fashionpedia)
    python -m src.preprocessing.fashionpedia_preprocess "${PASSTHROUGH[@]}"
    ;;
  people_clothing_segmentation|people-clothing-segmentation|pcs)
    python -m src.preprocessing.people_clothing_segmentation_preprocess "${PASSTHROUGH[@]}"
    ;;
  "")
    echo "Missing required option: --dataset-name" >&2
    exit 2
    ;;
  *)
    echo "Unsupported dataset: ${DATASET_NAME}" >&2
    exit 2
    ;;
esac

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DATASET="${1:-aliccp}"
REGION="${2:-ES}"

SEED="${SEED:-0}"
MODEL_NAME="${MODEL_NAME:-MMOE}"
LOSS_TYPE="${LOSS_TYPE:-ips}"
SPLIT="${SPLIT:-val}"
OUTPUT_ROOT="${OUTPUT_ROOT:-analysis/propensity_proxy_bias}"
PROFILE_BINS="${PROFILE_BINS:-20,50}"
FEATURE_SAMPLE_ROWS="${FEATURE_SAMPLE_ROWS:-1000000}"
MAX_DENSE_FEATURES="${MAX_DENSE_FEATURES:-12}"
MAX_SPARSE_FEATURES="${MAX_SPARSE_FEATURES:-20}"
MIN_COHORT_N="${MIN_COHORT_N:-1000}"
TOPK_VALUES="${TOPK_VALUES:-20}"
DENSE_BINS="${DENSE_BINS:-5}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

"$PYTHON_BIN" models/analyze_propensity_proxy_bias.py \
  --dataset "$DATASET" \
  --region "$REGION" \
  --seed "$SEED" \
  --model_name "$MODEL_NAME" \
  --loss_type "$LOSS_TYPE" \
  --split "$SPLIT" \
  --output_root "$OUTPUT_ROOT" \
  --profile_bins "$PROFILE_BINS" \
  --feature_sample_rows "$FEATURE_SAMPLE_ROWS" \
  --max_dense_features "$MAX_DENSE_FEATURES" \
  --max_sparse_features "$MAX_SPARSE_FEATURES" \
  --min_cohort_n "$MIN_COHORT_N" \
  --topk_values "$TOPK_VALUES" \
  --dense_bins "$DENSE_BINS" \
  $EXTRA_ARGS

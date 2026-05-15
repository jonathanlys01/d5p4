#!/usr/bin/env bash

set -euo pipefail

ESMC_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
D3P2_ROOT="$(cd -- "${ESMC_ROOT}/../../.." && pwd)"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  else
    echo "Neither python nor python3 is available on PATH." >&2
    exit 1
  fi
fi

DEFAULT_CONFIG_PATH="${D3P2_ROOT}/src/d5p4/_default.yaml"

strip_quotes() {
  local value="$1"
  value="${value%\"}"
  value="${value#\"}"
  value="${value%\'}"
  value="${value#\'}"
  printf '%s\n' "$value"
}

read_default_value() {
  local key="$1"
  local line
  line="$(grep -E "^${key}:" "${DEFAULT_CONFIG_PATH}" | head -n 1 || true)"
  if [[ -z "$line" ]]; then
    echo "Missing ${key} in ${DEFAULT_CONFIG_PATH}." >&2
    exit 1
  fi
  strip_quotes "${line#*: }"
}

resolve_env_path_or() {
  local spec="$1"

  if [[ "$spec" =~ ^\$\{env_path_or:([^,]+),([^,]+),(.+)\}$ ]]; then
    local env_name="${BASH_REMATCH[1]}"
    local suffix="${BASH_REMATCH[2]}"
    local fallback="${BASH_REMATCH[3]}"
    local env_value="${!env_name:-}"

    if [[ -n "$env_value" ]]; then
      printf '%s\n' "${env_value%/}/${suffix}"
    else
      printf '%s\n' "$fallback"
    fi
    return
  fi

  printf '%s\n' "$spec"
}

DEFAULT_MDLM_PATH="$(resolve_env_path_or "$(read_default_value mdlm_model_path)")"
DEFAULT_MDLM_TOKENIZER="$(resolve_env_path_or "$(read_default_value mdlm_tokenizer)")"
DEFAULT_PPL_MODEL_PATH="$(resolve_env_path_or "$(read_default_value ppl_model_id)")"
DEFAULT_CACHE_DIR="$(resolve_env_path_or "$(read_default_value cache_dir)")"

make_absolute_dir() {
  local value="$1"
  if [[ "$value" = /* ]]; then
    printf '%s\n' "$value"
  else
    printf '%s\n' "${D3P2_ROOT}/${value#./}"
  fi
}

# Main ESMC knobs.
LAMBDA_WEIGHT="${LAMBDA_WEIGHT:-5.0}"
NUM_PARTICLES="${NUM_PARTICLES:-8}"
RESAMPLE_INTERVAL="${RESAMPLE_INTERVAL:-50}"
STEPS="${STEPS:-128}"
NUM_RUNS="${NUM_RUNS:-8}"
SEED="${SEED:-1}"

# Model and runtime paths. Override these if you want a different local asset.
MDLM_CHECKPOINT_PATH="${MDLM_CHECKPOINT_PATH:-$DEFAULT_MDLM_PATH}"
MDLM_TOKENIZER_PATH="${MDLM_TOKENIZER_PATH:-$DEFAULT_MDLM_TOKENIZER}"
PPL_MODEL_PATH="${PPL_MODEL_PATH:-$DEFAULT_PPL_MODEL_PATH}"
CACHE_DIR="${CACHE_DIR:-$DEFAULT_CACHE_DIR}"
CACHE_DIR="$(make_absolute_dir "$CACHE_DIR")"

MODEL_LENGTH="${MODEL_LENGTH:-1024}"
POTENTIAL_TYPE="${POTENTIAL_TYPE:-max}"
PREDICTOR="${PREDICTOR:-ddpm_cache}"
ACCELERATOR="${ACCELERATOR:-cuda}"
DEVICES="${DEVICES:-1}"
DRY_RUN="${DRY_RUN:-0}"
RUN_NAME="${RUN_NAME:-smc_mdlm}"
OUTPUT_DIR="${OUTPUT_DIR:-${ESMC_ROOT}/outputs/smc/${RUN_NAME}}"

export HF_HOME="${HF_HOME:-${CACHE_DIR}/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}}"

if [[ -z "${BACKBONE:-}" ]]; then
  if [[ "${MDLM_CHECKPOINT_PATH}" == *.ckpt ]]; then
    BACKBONE="dit"
  else
    BACKBONE="mdlm_ref"
  fi
fi

echo "esmc_ref root: ${ESMC_ROOT}"
echo "Using MDLM path: ${MDLM_CHECKPOINT_PATH}"
echo "Using tokenizer path: ${MDLM_TOKENIZER_PATH}"
echo "Using eval LM path: ${PPL_MODEL_PATH}"
echo "Output dir: ${OUTPUT_DIR}"

cd "${ESMC_ROOT}"

cmd=(
  "${PYTHON_BIN}"
  main.py
  "sampling=smc"
  "seed=${SEED}"
  "backbone=${BACKBONE}"
  "model.length=${MODEL_LENGTH}"
  "data.tokenizer_name_or_path=${MDLM_TOKENIZER_PATH}"
  "data.cache_dir=${CACHE_DIR}"
  "eval.checkpoint_path=${MDLM_CHECKPOINT_PATH}"
  "eval.gen_ppl_eval_model_name_or_path=${PPL_MODEL_PATH}"
  "sampling.predictor=${PREDICTOR}"
  "sampling.steps=${STEPS}"
  "sampling.num_sample_batches=${NUM_RUNS}"
  "smc.num_particles=${NUM_PARTICLES}"
  "smc.resample_interval=${RESAMPLE_INTERVAL}"
  "smc.lambda_weight=${LAMBDA_WEIGHT}"
  "smc.potential_type=${POTENTIAL_TYPE}"
  "trainer.accelerator=${ACCELERATOR}"
  "trainer.devices=${DEVICES}"
  "hydra.run.dir=${OUTPUT_DIR}"
)

printf 'Running command:\n'
printf '  %q' "${cmd[@]}"
printf '\n'

if [[ "${DRY_RUN}" == "1" ]]; then
  exit 0
fi

"${cmd[@]}"

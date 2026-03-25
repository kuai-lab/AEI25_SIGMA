#!/usr/bin/env bash
# Orchestrate the full ArtGS pipeline from coarse -> align -> per-state render -> predict -> train -> render/eval.
set -euo pipefail

DATASET="artgs"
SUBSET="sapien"
SCENES=""
STAGES="coarse,align,per_state_render,predict,train,render_video,eval"
CUDA_DEVICES=""
SKIP_EXISTING=0
DRY_RUN=0

COARSE_MODEL_NAME="coarse_gs"
COARSE_ITER=10000
COARSE_RES=2
COARSE_OPACITY_REG=0.1

PREDICT_MODEL_NAME="pred"
PREDICT_ITER=5000
PREDICT_RES=8
PREDICT_DENSIFY=0.001

PER_STATE_RES=1
PER_STATE_ITER=50
PER_STATE_DENSIFY=0.001

TRAIN_MODEL_NAME="artgs"
TRAIN_ITER=20000
TRAIN_RES=1
TRAIN_DENSIFY=0.001
USE_ART_TYPE_PRIOR=1

RENDER_RES=1
RENDER_ITER="best"
RENDER_FRAMES=30
RENDER_WHITE_BG=1

EVAL_RES=1
EVAL_ITER="best"

RANDOM_BG=1

usage() {
  cat <<EOF
Usage: bash scripts/pipeline.sh --dataset artgs --subset sapien --scenes excavator[,lamp] [options]

Options:
  --dataset NAME              Dataset name (default: ${DATASET})
  --subset NAME               Subset name (default: ${SUBSET})
  --scenes LIST               Comma-separated scene list (required)
  --stages LIST               Comma-separated stages (default: ${STAGES})
                              Allowed: coarse,align,per_state_render,predict,train,render_video,eval,all
  --cuda DEVICES              Set CUDA_VISIBLE_DEVICES
  --skip-existing             Skip stages if key outputs already exist (coarse/align)
  --dry-run                   Print commands without running them
  --no-random-bg              Disable random background rendering flags
  --no-art-type-prior         Disable art type prior during training
EOF
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset) DATASET="$2"; shift 2;;
    --subset) SUBSET="$2"; shift 2;;
    --scenes) SCENES="$2"; shift 2;;
    --stages) STAGES="$2"; shift 2;;
    --cuda) CUDA_DEVICES="$2"; shift 2;;
    --skip-existing) SKIP_EXISTING=1; shift;;
    --dry-run) DRY_RUN=1; shift;;
    --no-random-bg) RANDOM_BG=0; shift;;
    --random-bg) RANDOM_BG=1; shift;;
    --no-art-type-prior) USE_ART_TYPE_PRIOR=0; shift;;
    --use-art-type-prior) USE_ART_TYPE_PRIOR=1; shift;;
    -h|--help) usage;;
    *) echo "Unknown argument: $1" >&2; usage;;
  esac
done

if [[ -z "${SCENES}" ]]; then
  echo "Error: --scenes is required." >&2
  usage
fi

IFS=',' read -r -a SCENE_LIST <<< "${SCENES}"
IFS=',' read -r -a STAGE_LIST <<< "${STAGES}"
STAGE_ORDER=(coarse align per_state_render predict train render_video eval)

if [[ -n "${CUDA_DEVICES}" ]]; then
  export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
fi

run_cmd() {
  echo "[cmd] $*"
  if [[ "${DRY_RUN}" -eq 0 ]]; then
    "$@"
  fi
}

stage_enabled() {
  local target="$1"
  for s in "${STAGE_LIST[@]}"; do
    if [[ "$s" == "$target" || "$s" == "all" ]]; then
      return 0
    fi
  done
  return 1
}

random_bg_flag() {
  [[ "${RANDOM_BG}" -eq 1 ]] && echo "--random_bg_color" || true
}

art_prior_flag() {
  [[ "${USE_ART_TYPE_PRIOR}" -eq 1 ]] && echo "--use_art_type_prior" || true
}

white_bg_flag() {
  [[ "${RENDER_WHITE_BG}" -eq 1 ]] && echo "--white_background" || true
}

for scene in "${SCENE_LIST[@]}"; do
  echo "=== Scene: ${scene} ==="
  coarse_dir="outputs/${DATASET}/${SUBSET}/${scene}/${COARSE_MODEL_NAME}/point_cloud/iteration_${COARSE_ITER}"
  coarse_ply="${coarse_dir}/gaussian_0.ply"
  canonical_ply="${coarse_dir}/point_cloud.ply"

  for stage in "${STAGE_ORDER[@]}"; do
    stage_enabled "${stage}" || continue
    echo "-- Stage: ${stage}"

    case "${stage}" in
      coarse)
        if [[ "${SKIP_EXISTING}" -eq 1 && -f "${coarse_ply}" ]]; then
          echo "[skip] coarse exists at ${coarse_ply}"
          continue
        fi
        run_cmd python train_coarse.py \
          --dataset "${DATASET}" \
          --subset "${SUBSET}" \
          --scene_name "${scene}" \
          --model_path "outputs/${DATASET}/${SUBSET}/${scene}/${COARSE_MODEL_NAME}" \
          --resolution "${COARSE_RES}" \
          --iterations "${COARSE_ITER}" \
          --opacity_reg_weight "${COARSE_OPACITY_REG}" \
          $(random_bg_flag)
        ;;

      align)
        if [[ "${SKIP_EXISTING}" -eq 1 && -f "${canonical_ply}" ]]; then
          echo "[skip] alignment exists at ${canonical_ply}"
          continue
        fi
        run_cmd python learnable_scaling_norm.py \
          --dataset "${DATASET}" \
          --subset "${SUBSET}" \
          --scene_name "${scene}" \
          --coarse_name "${COARSE_MODEL_NAME}" \
          --iteration "${COARSE_ITER}"
        run_cmd python get_coarse_gs_w_scale.py \
          --dataset "${DATASET}" \
          --subset "${SUBSET}" \
          --scene_name "${scene}" \
          --coarse_name "${COARSE_MODEL_NAME}" \
          --iteration "${COARSE_ITER}"
        ;;

      per_state_render)
        run_cmd python per_state_rendering.py \
          --dataset "${DATASET}" \
          --subset "${SUBSET}" \
          --scene_name "${scene}" \
          --model_path "outputs/${DATASET}/${SUBSET}/${scene}/${PREDICT_MODEL_NAME}" \
          --resolution "${PER_STATE_RES}" \
          --iterations "${PER_STATE_ITER}" \
          --densify_grad_threshold "${PER_STATE_DENSIFY}" \
          --coarse_name "${COARSE_MODEL_NAME}" \
          --eval \
          $(random_bg_flag)
        ;;

      predict)
        run_cmd python train_predict.py \
          --dataset "${DATASET}" \
          --subset "${SUBSET}" \
          --scene_name "${scene}" \
          --model_path "outputs/${DATASET}/${SUBSET}/${scene}/${PREDICT_MODEL_NAME}" \
          --resolution "${PREDICT_RES}" \
          --iterations "${PREDICT_ITER}" \
          --densify_grad_threshold "${PREDICT_DENSIFY}" \
          --coarse_name "${COARSE_MODEL_NAME}" \
          --eval \
          $(random_bg_flag)
        ;;

      train)
        run_cmd python train.py \
          --dataset "${DATASET}" \
          --subset "${SUBSET}" \
          --scene_name "${scene}" \
          --model_path "outputs/${DATASET}/${SUBSET}/${scene}/${TRAIN_MODEL_NAME}" \
          --resolution "${TRAIN_RES}" \
          --iterations "${TRAIN_ITER}" \
          --coarse_name "${COARSE_MODEL_NAME}" \
          --seed 0 \
          --densify_grad_threshold "${TRAIN_DENSIFY}" \
          --eval \
          $(art_prior_flag) \
          $(random_bg_flag)
        ;;

      render_video)
        run_cmd python render_video.py \
          --dataset "${DATASET}" \
          --subset "${SUBSET}" \
          --scene_name "${scene}" \
          --model_path "outputs/${DATASET}/${SUBSET}/${scene}/${TRAIN_MODEL_NAME}" \
          --resolution "${RENDER_RES}" \
          --iteration "${RENDER_ITER}" \
          --N_frames "${RENDER_FRAMES}" \
          $(white_bg_flag)
        ;;

      eval)
        run_cmd python render.py \
          --dataset "${DATASET}" \
          --subset "${SUBSET}" \
          --scene_name "${scene}" \
          --model_path "outputs/${DATASET}/${SUBSET}/${scene}/${TRAIN_MODEL_NAME}" \
          --resolution "${EVAL_RES}" \
          --iteration "${EVAL_ITER}" \
          --skip_test
        ;;
    esac
  done
done

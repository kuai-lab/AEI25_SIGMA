#!/usr/bin/env python3
"""
Orchestrates the full ArtGS pipeline so individual steps don't have to be run manually.

Stages (in order):
    1) coarse           -> train coarse Gaussians
    2) align            -> learn scaling/translation and build canonical Gaussians
    3) per_state_render -> render per-state RGB-D used by the predictor
    4) predict          -> joint type prediction
    5) train            -> full ArtGS optimization
    6) render_video     -> render a short video for quick inspection
    7) eval             -> evaluation/rendering of the final checkpoint
"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path
from typing import Iterable, List


STAGE_ORDER = [
    "coarse",
    "align",
    "per_state_render",
    "predict",
    "train",
    "render_video",
    "eval",
]


def _run(cmd: List[str], env: dict, dry_run: bool):
    cmd_str = " ".join(cmd)
    print(f"\n[cmd] {cmd_str}")
    if dry_run:
        return
    subprocess.run(cmd, check=True, env=env)


def _build_output_root(dataset: str, subset: str, scene: str) -> Path:
    return Path("outputs") / dataset / subset / scene


def _build_coarse_dir(dataset: str, subset: str, scene: str, coarse_name: str, iteration: int) -> Path:
    return (
        _build_output_root(dataset, subset, scene)
        / coarse_name
        / "point_cloud"
        / f"iteration_{iteration}"
    )


def stage_coarse(args, scene: str, env: dict):
    coarse_dir = _build_coarse_dir(args.dataset, args.subset, scene, args.coarse_model_name, args.coarse_iterations)
    expected = coarse_dir / "gaussian_0.ply"
    if args.skip_existing and expected.exists():
        print(f"[skip] coarse exists at {expected}")
        return

    cmd = [
        "python",
        "train_coarse.py",
        "--dataset",
        args.dataset,
        "--subset",
        args.subset,
        "--scene_name",
        scene,
        "--model_path",
        str(_build_output_root(args.dataset, args.subset, scene) / args.coarse_model_name),
        "--resolution",
        str(args.coarse_resolution),
        "--iterations",
        str(args.coarse_iterations),
        "--opacity_reg_weight",
        str(args.coarse_opacity_reg_weight),
    ]
    if args.random_bg_color:
        cmd.append("--random_bg_color")
    _run(cmd, env, args.dry_run)


def stage_align(args, scene: str, env: dict):
    coarse_dir = _build_coarse_dir(args.dataset, args.subset, scene, args.coarse_model_name, args.coarse_iterations)
    canonical_marker = coarse_dir / "point_cloud.ply"
    if args.skip_existing and canonical_marker.exists():
        print(f"[skip] alignment exists at {canonical_marker}")
        return

    align_common = [
        "--dataset",
        args.dataset,
        "--subset",
        args.subset,
        "--scene_name",
        scene,
        "--coarse_name",
        args.coarse_model_name,
        "--iteration",
        str(args.coarse_iterations),
    ]
    _run(["python", "learnable_scaling_norm.py", *align_common], env, args.dry_run)
    _run(["python", "get_coarse_gs_w_scale.py", *align_common], env, args.dry_run)


def stage_per_state_render(args, scene: str, env: dict):
    cmd = [
        "python",
        "per_state_rendering.py",
        "--dataset",
        args.dataset,
        "--subset",
        args.subset,
        "--scene_name",
        scene,
        "--model_path",
        str(_build_output_root(args.dataset, args.subset, scene) / args.predict_model_name),
        "--resolution",
        str(args.per_state_resolution),
        "--iterations",
        str(args.per_state_iterations),
        "--densify_grad_threshold",
        str(args.per_state_densify_grad_threshold),
        "--coarse_name",
        args.coarse_model_name,
        "--eval",
    ]
    if args.random_bg_color:
        cmd.append("--random_bg_color")
    _run(cmd, env, args.dry_run)


def stage_predict(args, scene: str, env: dict):
    cmd = [
        "python",
        "train_predict.py",
        "--dataset",
        args.dataset,
        "--subset",
        args.subset,
        "--scene_name",
        scene,
        "--model_path",
        str(_build_output_root(args.dataset, args.subset, scene) / args.predict_model_name),
        "--resolution",
        str(args.predict_resolution),
        "--iterations",
        str(args.predict_iterations),
        "--densify_grad_threshold",
        str(args.predict_densify_grad_threshold),
        "--coarse_name",
        args.coarse_model_name,
        "--eval",
    ]
    if args.random_bg_color:
        cmd.append("--random_bg_color")
    _run(cmd, env, args.dry_run)


def stage_train(args, scene: str, env: dict):
    cmd = [
        "python",
        "train.py",
        "--dataset",
        args.dataset,
        "--subset",
        args.subset,
        "--scene_name",
        scene,
        "--model_path",
        str(_build_output_root(args.dataset, args.subset, scene) / args.train_model_name),
        "--resolution",
        str(args.train_resolution),
        "--iterations",
        str(args.train_iterations),
        "--coarse_name",
        args.coarse_model_name,
        "--seed",
        str(args.seed),
        "--densify_grad_threshold",
        str(args.train_densify_grad_threshold),
        "--eval",
    ]
    if args.use_art_type_prior:
        cmd.append("--use_art_type_prior")
    if args.random_bg_color:
        cmd.append("--random_bg_color")
    _run(cmd, env, args.dry_run)


def stage_render_video(args, scene: str, env: dict):
    cmd = [
        "python",
        "render_video.py",
        "--dataset",
        args.dataset,
        "--subset",
        args.subset,
        "--scene_name",
        scene,
        "--model_path",
        str(_build_output_root(args.dataset, args.subset, scene) / args.train_model_name),
        "--resolution",
        str(args.render_resolution),
        "--iteration",
        str(args.render_iteration),
        "--N_frames",
        str(args.render_frames),
    ]
    if args.render_white_background:
        cmd.append("--white_background")
    _run(cmd, env, args.dry_run)


def stage_eval(args, scene: str, env: dict):
    cmd = [
        "python",
        "render.py",
        "--dataset",
        args.dataset,
        "--subset",
        args.subset,
        "--scene_name",
        scene,
        "--model_path",
        str(_build_output_root(args.dataset, args.subset, scene) / args.train_model_name),
        "--resolution",
        str(args.eval_resolution),
        "--iteration",
        str(args.eval_iteration),
        "--skip_test",
    ]
    _run(cmd, env, args.dry_run)


STAGE_MAP = {
    "coarse": stage_coarse,
    "align": stage_align,
    "per_state_render": stage_per_state_render,
    "predict": stage_predict,
    "train": stage_train,
    "render_video": stage_render_video,
    "eval": stage_eval,
}


def _resolve_stages(requested: Iterable[str]) -> List[str]:
    req = list(requested)
    if "all" in req:
        return STAGE_ORDER
    unknown = [s for s in req if s not in STAGE_MAP]
    if unknown:
        raise ValueError(f"Unknown stages requested: {unknown}")
    return [s for s in STAGE_ORDER if s in req]


def parse_args():
    parser = argparse.ArgumentParser(description="Run the full ArtGS pipeline without manual stage switching.")
    parser.add_argument("--dataset", default="artgs")
    parser.add_argument("--subset", default="sapien")
    parser.add_argument("--scenes", nargs="+", required=True, help="One or more scene names to process.")
    parser.add_argument("--stages", nargs="+", default=["all"], choices=STAGE_ORDER + ["all"], help="Stages to run.")
    parser.add_argument("--cuda", default=None, help="Value for CUDA_VISIBLE_DEVICES. Leave unset to use current env.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip stages when their key outputs already exist.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--coarse-model-name", default="coarse_gs")
    parser.add_argument("--coarse-iterations", type=int, default=10000)
    parser.add_argument("--coarse-resolution", type=int, default=2)
    parser.add_argument("--coarse-opacity-reg-weight", type=float, default=0.1)

    parser.add_argument("--predict-model-name", default="pred")
    parser.add_argument("--predict-iterations", type=int, default=5000)
    parser.add_argument("--predict-resolution", type=int, default=8)
    parser.add_argument("--predict-densify-grad-threshold", type=float, default=0.001)

    parser.add_argument("--per-state-resolution", type=int, default=1)
    parser.add_argument("--per-state-iterations", type=int, default=50)
    parser.add_argument("--per-state-densify-grad-threshold", type=float, default=0.001)

    parser.add_argument("--train-model-name", default="artgs")
    parser.add_argument("--train-iterations", type=int, default=20000)
    parser.add_argument("--train-resolution", type=int, default=1)
    parser.add_argument("--train-densify-grad-threshold", type=float, default=0.001)
    parser.add_argument("--use-art-type-prior", dest="use_art_type_prior", action="store_true", default=True)
    parser.add_argument("--no-art-type-prior", dest="use_art_type_prior", action="store_false")

    parser.add_argument("--render-resolution", type=int, default=1)
    parser.add_argument("--render-iteration", default="best")
    parser.add_argument("--render-frames", type=int, default=30)
    parser.add_argument("--render-white-background", dest="render_white_background", action="store_true", default=True)
    parser.add_argument("--render-colored-background", dest="render_white_background", action="store_false")

    parser.add_argument("--eval-resolution", type=int, default=1)
    parser.add_argument("--eval-iteration", default="best")

    parser.add_argument("--random-bg", dest="random_bg_color", action="store_true", default=True)
    parser.add_argument("--no-random-bg", dest="random_bg_color", action="store_false")
    return parser.parse_args()


def main():
    args = parse_args()
    stages = _resolve_stages(args.stages)
    env = os.environ.copy()
    if args.cuda is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.cuda)

    print(f"Running stages: {stages}")
    for scene in args.scenes:
        print(f"\n=== Scene: {scene} ===")
        for stage in stages:
            print(f"\n-- Stage: {stage}")
            STAGE_MAP[stage](args, scene, env)


if __name__ == "__main__":
    main()

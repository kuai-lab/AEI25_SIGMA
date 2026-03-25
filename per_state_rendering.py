#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import json
import torch
import copy
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render
import sys
from scene import Scene, GaussianModel, DeformModel
from utils.general_utils import safe_state
import tqdm
import cv2
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams
from pytorch_lightning import seed_everything
from utils.metrics import *
import torchvision
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt
import time


class Trainer:
    def __init__(self, args, dataset, opt, pipe, larger_state) -> None:
        self.dataset = dataset
        self.args = args
        self.opt = opt
        self.pipe = pipe
        
        self.gaussians = [GaussianModel(dataset.sh_degree), GaussianModel(dataset.sh_degree)]
        self.static_gaussian = GaussianModel(dataset.sh_degree)
        self.scene = Scene(dataset, self.gaussians, load_iteration=None)

        p = args.source_path.replace('data/', 'outputs/')
        coarse_name = self.args.coarse_name
        self.xyzs = self.gaussians[0].load_ply(f'{p}/{coarse_name}/point_cloud/iteration_10000/gaussian_0.ply')
        self.xyzs = self.gaussians[1].load_ply(f'{p}/{coarse_name}/point_cloud/iteration_10000/gaussian_1_transformed.ply')
        print('Init canonical gaussians from coarse gaussian.')

        self.gaussians[0].training_setup(opt)
        self.gaussians[1].training_setup(opt)

        if larger_state == 0:
            self.static_gaussian = copy.deepcopy(self.gaussians[0])
        else:
            self.static_gaussian = copy.deepcopy(self.gaussians[1])

        # get static mask
        mask_path = f'{p}/{coarse_name}/point_cloud/iteration_10000/mask_static.npy'
        if os.path.exists(mask_path):
            mask = np.load(mask_path)
            mask = torch.tensor(mask, dtype=torch.bool, device="cuda")
            mask = ~mask

        self.static_gaussian.prune_points(mask)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        self.viewpoint_stacks = [self.scene.getTrainCameras_start(), self.scene.getTrainCameras_end()]
        self.viewpoint_statics = self.scene.getTrainCameras_start()


    def train(self):
        for state in range(2):
            # p = self.args.source_path.replace('data/', 'outputs/')
            p = self.args.source_path
            if state == 0:
                s = 'start'
            else:
                s = 'end'
            rgba_path = f'{p}/{s}/train/rgba'
            depth_path = f'{p}/{s}/train/depth'
            static_path = f'{p}/static/train/rgba'
            static_depth_path = f'{p}/static/train/depth'
            os.makedirs(rgba_path, exist_ok=True)
            os.makedirs(depth_path, exist_ok=True)
            os.makedirs(static_path, exist_ok=True)
            os.makedirs(static_depth_path, exist_ok=True)

            for id in range(len(self.viewpoint_stacks[state])):
                viewpoint_cam = self.viewpoint_stacks[state][id]

                # Render
                random_bg = False
                bg = self.background if not random_bg else torch.rand_like(self.background).cuda()
                d_xyz, d_rot = None, None
                render_pkg_re = render(viewpoint_cam, self.gaussians[state], self.pipe, bg, d_xyz=d_xyz, d_rot=d_rot)
                image, viewspace_point_tensor, visibility_filter, radii = render_pkg_re["render"], render_pkg_re["viewspace_points"], render_pkg_re["visibility_filter"], render_pkg_re["radii"]
                depth = render_pkg_re["depth"]
                depth = depth[0].cpu().detach().numpy()

                alpha = render_pkg_re["alpha"]
                image = torch.cat([image, alpha], dim=0)

                # 만약 원 코드의 마스크 적용과 클리핑 처리가 필요하다면:
                # (예시에서는 0.1 미만은 0 처리)
                depth[depth < 0.1] = 0
                # 원 코드에서는 저장 전에 1000으로 나눴으므로, 반대로 1000을 곱해줌
                depth_scaled = depth * 1e3
                # float 값들을 uint16으로 캐스팅
                depth_uint16 = depth_scaled.astype(np.uint16)

                # ====== render static gaussian ======
                render_static = render(viewpoint_cam, self.static_gaussian, self.pipe, bg, d_xyz=d_xyz, d_rot=d_rot)
                static_image, static_depth = render_static["render"], render_static["depth"]
                static_depth = static_depth[0].cpu().detach().numpy()
                static_depth[static_depth < 0.1] = 0
                static_depth_scaled = static_depth * 1e3
                static_depth_uint16 = static_depth_scaled.astype(np.uint16)

                static_alpha = render_static["alpha"]
                static_image = torch.cat([static_image, static_alpha], dim=0)

                # save image
                # torchvision.utils.save_image(image, f'{rgba_path}/{id:04d}.png')
                # torchvision.utils.save_image(static_image, f'{static_path}/{id:04d}.png')
                # torchvision.utils.save_image(depth, f'{depth_path}/{id:04d}.png')
                image_pil = to_pil_image(image)
                static_image_pil = to_pil_image(static_image)
                image_pil.save(f'{rgba_path}/{id:04d}.png')
                static_image_pil.save(f'{static_path}/{id:04d}.png')

                # Save depth as .png using OpenCV
                cv2.imwrite(f'{depth_path}/{id:04d}.png', depth_uint16)
                cv2.imwrite(f'{static_depth_path}/{id:04d}.png', static_depth_uint16)
                print(f"Saved image {id} for state {s}")
        print("All images saved.")



if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    
    args = parser.parse_args(sys.argv[1:])
    args.source_path = f"{args.source_path}/{args.dataset}/{args.subset}/{args.scene_name}"

    file = json.load(open('arguments/larger_motion_state.json', 'r'))
    larger_motion_state = file[args.dataset][args.subset][args.scene_name]
    print(f"larger_motion_state: {larger_motion_state}")

    trainer = Trainer(args=args, dataset=lp.extract(args), opt=op.extract(args), pipe=pp.extract(args), larger_state=larger_motion_state)
    trainer.train()
    # print("\nTraining complete.")

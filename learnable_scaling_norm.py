import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import open3d as o3d
from pathlib import Path
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils.pointnet2_utils import farthest_point_sample, index_points
from pytorch3d.loss import chamfer_distance



class PointCloudScaling(nn.Module):
    def __init__(self):
        super(PointCloudScaling, self).__init__()
        # randomly initialize the scaling factors
        # self.x_scale = nn.Parameter(torch.randn(1)[0] * 0.1 + 1.0)
        # self.y_scale = nn.Parameter(torch.randn(1)[0] * 0.1 + 1.0)
        # self.z_scale = nn.Parameter(torch.randn(1)[0] * 0.1 + 1.0)

        self.x_scale = nn.Parameter(torch.tensor(1.0))
        self.y_scale = nn.Parameter(torch.tensor(1.0))
        self.z_scale = nn.Parameter(torch.tensor(1.0))

        # translate
        # self.x_trans = nn.Parameter(torch.randn(1)[0] * 0.1)    
        # self.y_trans = nn.Parameter(torch.randn(1)[0] * 0.1)
        # self.z_trans = nn.Parameter(torch.randn(1)[0] * 0.1)

        self.x_trans = nn.Parameter(torch.tensor(0.0))    
        self.y_trans = nn.Parameter(torch.tensor(0.0))
        self.z_trans = nn.Parameter(torch.tensor(0.0))


    def forward(self, pcd):
        # Apply scaling
        scaling_factors = torch.stack([self.x_scale, self.y_scale, self.z_scale])
        pcd_scaled = pcd * scaling_factors

        # Apply translation
        translation_factor = torch.stack([self.x_trans, self.y_trans, self.z_trans])
        pcd_scaled += translation_factor
        return pcd_scaled
    
class PointCloudScalingNoise(nn.Module):
    def __init__(self,
                 noise_frac: float = 0.01,     # 축별 범위 대비 노이즈 표준편차 비율 (기본 1%)
                 noise_prob: float = 1.0,      # 노이즈 적용 확률 (기본 항상 적용)
                 enable_noise: bool = True,    # 노이즈 토글
                 noise_clip_sigma: float = 3.0,# 과도한 샘플 클램핑(±3σ)
                 apply_noise_in_eval: bool = True # eval 모드에서도 적용할지
                 ):
        super(PointCloudScalingNoise, self).__init__()

        # learnable scale/translation
        self.x_scale = nn.Parameter(torch.tensor(1.0))
        self.y_scale = nn.Parameter(torch.tensor(1.0))
        self.z_scale = nn.Parameter(torch.tensor(1.0))

        self.x_trans = nn.Parameter(torch.tensor(0.0))    
        self.y_trans = nn.Parameter(torch.tensor(0.0))
        self.z_trans = nn.Parameter(torch.tensor(0.0))

        # noise settings
        self.enable_noise = enable_noise
        self.noise_frac = float(noise_frac)
        self.noise_prob = float(noise_prob)
        self.noise_clip_sigma = float(noise_clip_sigma) if noise_clip_sigma is not None else None
        self.apply_noise_in_eval = bool(apply_noise_in_eval)

    def forward(self, pcd: torch.Tensor):
        """
        pcd: (N, 3)  — unscaled 입력 포인트 (여기에 대해 축별 range를 구해 노이즈 스케일을 정함)
        """
        # 1) scaling
        scaling_factors = torch.stack([self.x_scale, self.y_scale, self.z_scale])
        pcd_scaled = pcd * scaling_factors

        # 2) translation
        translation_factor = torch.stack([self.x_trans, self.y_trans, self.z_trans])
        pcd_scaled = pcd_scaled + translation_factor

        # 3) small random noise (axis-range-aware)
        if self.enable_noise and (self.training or self.apply_noise_in_eval):
            # 확률적으로만 적용하고 싶다면 noise_prob < 1.0 로 두면 됨
            if self.noise_prob >= 1.0 or torch.rand((), device=pcd.device) < self.noise_prob:
                # 입력 pcd(스케일 적용 전)의 축별 범위로 σ를 정함
                axis_range = (pcd.max(dim=0).values - pcd.min(dim=0).values).clamp_min(1e-8)  # (3,)
                std = axis_range * self.noise_frac                                         # (3,)
                # (N,3) 형태로 브로드캐스트하여 가우시안 노이즈 생성
                noise = torch.randn_like(pcd_scaled) * std

                # 과도한 노이즈 방지용 클램핑 (±kσ)
                if self.noise_clip_sigma is not None:
                    clip = std * self.noise_clip_sigma
                    # clip shape (3,) 를 (N,3)로 브로드캐스트
                    noise = torch.max(torch.min(noise,  clip), -clip)

                pcd_scaled = pcd_scaled + noise

        return pcd_scaled

# def chamfer_distance(pcd1, pcd2):
#     """
#     Compute the Chamfer Distance between two point clouds.
    
#     Args:
#         pcd1 (torch.Tensor): Tensor of shape (N, D) representing the first point cloud.
#         pcd2 (torch.Tensor): Tensor of shape (M, D) representing the second point cloud.

#     Returns:
#         torch.Tensor: The computed mean Chamfer Distance.
#     """
#     # Compute pairwise Euclidean distances between points in the two point clouds.
#     dist_matrix = torch.cdist(pcd1, pcd2, p=2)  # shape: (N, M)

#     min_dist_pcd1, _ = torch.min(dist_matrix, dim=1)
#     min_dist_pcd2, _ = torch.min(dist_matrix, dim=0)
#     return min_dist_pcd1.mean(), min_dist_pcd2.mean()


def add_point_noise_axis_scaled(points: torch.Tensor,
                                frac: float = 0.01,      # 축별 range의 1%를 표준편차로 사용
                                clip_sigma: float = 3.0, # ±3σ 클램핑
                                seed: int = 42):
    """
    points: (N,3) torch.Tensor
    frac:   std = axis_range * frac
    """
    assert points.dim() == 2 and points.size(-1) == 3
    device = points.device
    g = torch.Generator(device=device)
    if seed is not None:
        g.manual_seed(seed)

    axis_range = (points.max(0).values - points.min(0).values).clamp_min(1e-8)  # (3,)
    std = axis_range * frac                                                      # (3,)
    noise = torch.randn_like(points, generator=g) * std                          # (N,3)
    if clip_sigma is not None:
        clip = std * clip_sigma                                                  # (3,)
        noise = torch.max(torch.min(noise, clip), -clip)                         # 텐서 클램프

    return points + noise, noise

def voxelize_point_cloud(xyz, voxel_size):
    """
    Voxelize a point cloud by downsampling it to a grid of voxels.

    Args:
        xyz (tensor): Nx3 tensor of point cloud coordinates.
        voxel_size (float): Size of the voxel.

    Returns:
        tensor: Voxelized point cloud.
    """
    # Compute the voxel indices for each point
    voxel_indices = torch.floor(xyz / voxel_size).long()

    # Use a set to remove duplicate voxel indices
    unique_voxel_indices = torch.unique(voxel_indices, dim=0)

    # Compute the voxelized point cloud by averaging the points in each voxel
    voxelized_points = []
    for idx in unique_voxel_indices:
        mask = (voxel_indices == idx).all(dim=1)
        voxelized_points.append(xyz[mask].mean(dim=0))

    return torch.stack(voxelized_points)


def soft_voxelize(points, voxel_size, grid_size=None, origin=None, sigma=None):
    """
    Differentiable soft voxelization using a Gaussian kernel.
    
    Args:
        points (torch.Tensor): Shape (N,3) tensor of point cloud coordinates.
        voxel_size (float): Voxel size.
        grid_size (tuple or list, optional): Grid dimensions (nx, ny, nz). 
            If None, computed from points.
        origin (torch.Tensor or None): Origin of the grid. If None, computed as points.min(dim=0)[0].
        sigma (float, optional): Standard deviation for the Gaussian kernel.
            If None, set to voxel_size/2.
    
    Returns:
        occupancy_grid (torch.Tensor): Soft occupancy grid of shape (nx, ny, nz) with accumulated contributions.
        origin (torch.Tensor): The origin used for the voxel grid.
        grid_size (list): Grid size as [nx, ny, nz].
    """
    # Determine grid origin based on the minimum coordinate of points if not provided.
    if origin is None:
        origin = points.min(dim=0)[0]
    # Shift points relative to the origin.
    shifted_points = points - origin

    # If grid_size is not provided, compute it from the extents of the points.
    if grid_size is None:
        max_coords = shifted_points.max(dim=0)[0]
        grid_size = ((max_coords / voxel_size).ceil().long() + 1).tolist()
        # Add padding to the grid
        padding = 4
        nx, ny, nz = grid_size
        # Adjust the grid size to include padding
        nx += padding * 2
        ny += padding * 2
        nz += padding * 2

        # Update the grid size
        grid_size = [nx, ny, nz]
    nx, ny, nz = grid_size

    # Set Gaussian standard deviation if not provided.
    if sigma is None:
        sigma = voxel_size / 2.0

    device = points.device
    dtype = points.dtype

    # Create 1D arrays for each axis representing voxel centers.
    xs = (torch.arange(nx, device=device, dtype=dtype) + 0.5) * voxel_size
    ys = (torch.arange(ny, device=device, dtype=dtype) + 0.5) * voxel_size
    zs = (torch.arange(nz, device=device, dtype=dtype) + 0.5) * voxel_size

    # Create a meshgrid of voxel center coordinates.
    xx, yy, zz = torch.meshgrid(xs, ys, zs, indexing='ij')
    voxel_centers = torch.stack([xx, yy, zz], dim=-1)  # Shape: (nx, ny, nz, 3)

    # Flatten voxel centers to shape (nx*ny*nz, 3) for vectorized computation.
    voxel_centers_flat = voxel_centers.view(-1, 3)

    # Compute squared distances between each point and each voxel center.
    # Resulting shape: (N, nx*ny*nz)
    d2 = ((points.unsqueeze(1) - voxel_centers_flat.unsqueeze(0)) ** 2).sum(dim=2)

    # Compute Gaussian weights for each distance.
    weights = torch.exp(-d2 / (2 * sigma**2))

    # Sum over all points to obtain the soft occupancy for each voxel.
    occupancy_flat = weights.sum(dim=0)
    occupancy_grid = occupancy_flat.view(nx, ny, nz)

    return occupancy_grid, origin, grid_size


def apply_softmax_plane(tensor):
    """Flatten a 2D plane tensor, apply softmax, and reshape back."""
    shape = tensor.shape
    flat = tensor.reshape(-1)
    soft = F.softmax(flat, dim=0)
    return soft.reshape(shape)


def main(pcd1_path, pcd2_path):
    # set device 'mps'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load point clouds
    pcd1_o3d = o3d.io.read_point_cloud(pcd1_path)
    pcd2_o3d = o3d.io.read_point_cloud(pcd2_path)

    pcd1_xyz = np.asarray(pcd1_o3d.points)
    pcd2_xyz = np.asarray(pcd2_o3d.points)
    # pcd2_normal = np.asarray(pcd2_o3d.normals)


    # Convert to numpy arrays
    pcd1 = torch.tensor(np.asarray(pcd1_xyz), dtype=torch.float32).unsqueeze(0).to(device)
    pcd2 = torch.tensor(np.asarray(pcd2_xyz), dtype=torch.float32).unsqueeze(0).to(device)

    # farthest point sampling
    num_points = 512   # 512 was best for now
    pcd1_fps_idx = farthest_point_sample(pcd1, num_points)
    pcd2_fps_idx = farthest_point_sample(pcd2, num_points)
    pcd1_fps = index_points(pcd1, pcd1_fps_idx)[0]
    pcd2_fps = index_points(pcd2, pcd2_fps_idx)[0]


    iterations = 500
    learning_rate = 0.001

    voxel_sizes = [0.07, 0.05, 0.03]
    # voxel_sizes = [0.07, 0.05]
    scale_results = []
    trans_results = []
    # iterate over multiple voxel sizes
    for vs in voxel_sizes:
        print(f"Training with voxel size {vs}")
        voxel_size = vs
        # initialize a fresh model for this voxel size
        scaling_model = PointCloudScaling()
        # scaling_model = PointCloudScalingNoise()
        scaling_model.to(device)
        optimizer = torch.optim.AdamW(scaling_model.parameters(), lr=learning_rate)
        
        # existing training loop for this voxel size
        for i in tqdm(range(iterations)):
            loss = 0
            optimizer.zero_grad()

            # ========== forward pass =========================
            pcd2_scaled = scaling_model(pcd2_fps)
            all_points = torch.cat([pcd1_fps, pcd2_scaled], dim=0)
            min_all = all_points.min(dim=0)[0]
            pcd1_shifted = pcd1_fps - min_all
            pcd2_shifted = pcd2_scaled - min_all

            # ========== occupancy grid ==============
            occupancy_grid1, voxel_indices_min, grid_size_pcd1 = soft_voxelize(pcd1_shifted, voxel_size)
            occupancy_grid2, _, _ = soft_voxelize(pcd2_shifted, voxel_size, grid_size=grid_size_pcd1, origin=voxel_indices_min)
            
            # compute cumulative occupancy planes
            cumulative_z1 = torch.cumsum(occupancy_grid1, dim=2)
            cumulative_xy1 = cumulative_z1[:, :, -1]
            cumulative_x1 = torch.cumsum(occupancy_grid1, dim=0)
            cumulative_yz1 = cumulative_x1[-1, :, :]
            cumulative_y1 = torch.cumsum(occupancy_grid1, dim=1)
            cumulative_xz1 = cumulative_y1[:, -1, :]
            cumulative_z2 = torch.cumsum(occupancy_grid2, dim=2)
            cumulative_xy2 = cumulative_z2[:, :, -1]
            cumulative_x2 = torch.cumsum(occupancy_grid2, dim=0)
            cumulative_yz2 = cumulative_x2[-1, :, :]
            cumulative_y2 = torch.cumsum(occupancy_grid2, dim=1)
            cumulative_xz2 = cumulative_y2[:, -1, :]

            # ========== compute loss =========================
            grid_loss = (F.l1_loss(cumulative_xy1, cumulative_xy2) +
                         F.l1_loss(cumulative_yz1, cumulative_yz2) +
                         F.l1_loss(cumulative_xz1, cumulative_xz2))
            loss += grid_loss

            # ========== calc ymin bottom =========================
            threshold = 0.01
            pcd1_ymin = torch.max(pcd1_fps[:, 1])
            pcd2_ymin = torch.max(pcd2_scaled[:, 1])

            # bottom_indices = np.where(points[:, 1] <= ymin + threshold)[0]
            pcd1_bottom_indices = torch.where(pcd1_fps[:, 1] >= pcd1_ymin - threshold)[0]
            pcd2_bottom_indices = torch.where(pcd2_scaled[:, 1] >= pcd2_ymin - threshold)[0]

            # pcd1_ymin = torch.min(pcd1_fps[:, 1])
            # pcd2_ymin = torch.min(pcd2_scaled[:, 1])

            # # bottom_indices = np.where(points[:, 1] <= ymin + threshold)[0]
            # pcd1_bottom_indices = torch.where(pcd1_fps[:, 1] <= pcd1_ymin + threshold)[0]
            # pcd2_bottom_indices = torch.where(pcd2_scaled[:, 1] <= pcd2_ymin + threshold)[0]

            pcd1_bottom = pcd1_fps[pcd1_bottom_indices]
            pcd2_bottom = pcd2_scaled[pcd2_bottom_indices]


            # chamfer distance between bottom points
            pcd1_bottom = pcd1_bottom.unsqueeze(0)
            pcd2_bottom = pcd2_bottom.unsqueeze(0)

            cd, _ = chamfer_distance(pcd1_bottom, pcd2_bottom, batch_reduction=None, point_reduction=None)  # cd: [cd_start2end, cd_end2start]
            cd = cd[0].sum() + cd[1].sum() 
            loss += cd * 1.0

            loss.backward()
            optimizer.step()

            if i % 50 == 0:
                print(f"Iteration {i}, Loss: {loss.item()}, Grid Loss: {grid_loss.item()}, CD Loss: {cd.item()}")
        # after training for this voxel size, record results
        scale_results.append([scaling_model.x_scale.item(),
                              scaling_model.y_scale.item(),
                              scaling_model.z_scale.item()])
        trans_results.append([scaling_model.x_trans.item(),
                              scaling_model.y_trans.item(),
                              scaling_model.z_trans.item()])
    # end of voxel_sizes loop

    # compute mean scaling and translation
    mean_scales = np.mean(scale_results, axis=0)
    mean_trans = np.mean(trans_results, axis=0)
    x_scale, y_scale, z_scale = mean_scales
    x_trans, y_trans, z_trans = mean_trans

    # apply the learned scaling to pcd2
    # x_scale = scaling_model.x_scale.item()
    # y_scale = scaling_model.y_scale.item()
    # z_scale = scaling_model.z_scale.item()

    # apply the learned translation to pcd2
    # x_trans = scaling_model.x_trans.item()
    # y_trans = scaling_model.y_trans.item()
    # z_trans = scaling_model.z_trans.item()

    # scale pcd2
    pcd2_scaled = pcd2_xyz * np.array([x_scale, y_scale, z_scale])

    # translate pcd2
    pcd2_scaled = pcd2_scaled + (np.array([x_trans, y_trans, z_trans]))
    pcd2_scaled_o3d = o3d.geometry.PointCloud()
    pcd2_scaled_o3d.points = o3d.utility.Vector3dVector(pcd2_scaled)
    # pcd2_scaled_o3d.normals = o3d.utility.Vector3dVector(pcd2_normal)
    pcd2_scaled_o3d.paint_uniform_color([0, 1, 0])  # Green color

    pcd1_o3d = o3d.geometry.PointCloud()
    pcd1_o3d.points = o3d.utility.Vector3dVector(pcd1_xyz)
    pcd1_o3d.paint_uniform_color([0, 0, 0])  # Black color

    pcd2_o3d.paint_uniform_color([1, 0, 0])  # Red color

    
    # visualize the point clouds
    # o3d.visualization.webrtc_server.enable_webrtc()
    # o3d.visualization.draw([pcd1_o3d, pcd2_scaled_o3d, pcd2_o3d], show_ui=True)

    # save the scaled point cloud
    # o3d.io.write_point_cloud("plys/toy_drawer_end_scaled.ply", pcd2_scaled_o3d)
    # print(f"Scaled point cloud saved to 'plys/toy_drawer_end_scaled.ply'")

    # save the learned scaling and translation as .npy
    np.save(pcd1_path.replace('gaussian_0.ply', 'scaling.npy'), np.array([x_scale, y_scale, z_scale]))
    np.save(pcd1_path.replace('gaussian_0.ply', 'translation.npy'), np.array([x_trans, y_trans, z_trans]))
    # print(f"Learned scaling and translation saved to '{pcd1_path.replace('gaussian_0.ply', 'scaling.npy')}' and '{pcd1_path.replace('gaussian_0.ply', 'translation.npy')}'")
def _build_default_paths(dataset: str, subset: str, scene: str, coarse_name: str, iteration: int):
    base = Path("outputs") / dataset / subset / scene / coarse_name / "point_cloud" / f"iteration_{iteration}"
    pcd1 = base / "gaussian_0.ply"
    pcd2 = base / "gaussian_1.ply"
    return pcd1, pcd2


def _parse_args():
    parser = argparse.ArgumentParser(description="Learnable scaling and translation between two coarse Gaussians.")
    parser.add_argument("--dataset", default="artgs", help="Dataset name used to build default paths.")
    parser.add_argument("--subset", default="sapien", help="Subset name used to build default paths.")
    parser.add_argument("--scene_name", required=True, help="Scene name used to build default paths.")
    parser.add_argument("--coarse_name", default="coarse_gs", help="Coarse model directory name.")
    parser.add_argument("--iteration", type=int, default=10000, help="Iteration folder to read coarse Gaussians from.")
    parser.add_argument("--pcd1_path", type=Path, help="Path to gaussian_0.ply. Overrides dataset/subset/scene defaults.")
    parser.add_argument("--pcd2_path", type=Path, help="Path to gaussian_1.ply. Overrides dataset/subset/scene defaults.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    pcd1_path, pcd2_path = args.pcd1_path, args.pcd2_path

    if pcd1_path is None or pcd2_path is None:
        pcd1_path, pcd2_path = _build_default_paths(args.dataset, args.subset, args.scene_name, args.coarse_name, args.iteration)

    pcd1_path = Path(pcd1_path)
    pcd2_path = Path(pcd2_path)

    if not pcd1_path.exists():
        raise FileNotFoundError(f"Could not find coarse Gaussian at {pcd1_path}")
    if not pcd2_path.exists():
        raise FileNotFoundError(f"Could not find coarse Gaussian at {pcd2_path}")

    main(str(pcd1_path), str(pcd2_path))

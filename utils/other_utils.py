import torch
import numpy as np
from torch import nn
import open3d as o3d
from plyfile import PlyData
from sklearn.cluster import SpectralClustering
from scipy.optimize import linear_sum_assignment
from scipy.spatial import Delaunay
from pytorch3d.loss import chamfer_distance
from utils.pointnet2_utils import farthest_point_sample, index_points


def depth2normal(depth:torch.Tensor, focal:float=None):
    if depth.dim() == 2:
        depth = depth[None, None]
    elif depth.dim() == 3:
        depth = depth.squeeze()[None, None]
    if focal is None:
        focal = depth.shape[-1] / 2 / np.tan(torch.pi/6)
    depth = torch.cat([depth[:, :, :1], depth, depth[:, :, -1:]], dim=2)
    depth = torch.cat([depth[..., :1], depth, depth[..., -1:]], dim=3)
    kernel = torch.tensor([[[  0,   0,  0],
                            [-.5,   0, .5],
                            [  0,   0,  0]],
                           [[  0, -.5,  0],
                            [  0,   0,  0],
                            [  0,  .5,  0]]], device=depth.device, dtype=depth.dtype)[:, None]
    normal = torch.nn.functional.conv2d(depth, kernel, padding='valid')[0].permute(1, 2, 0)
    normal = normal / (depth[0, 0, 1:-1, 1:-1, None] + 1e-10) * focal
    normal = torch.cat([normal, torch.ones_like(normal[..., :1])], dim=-1)
    normal = normal / normal.norm(dim=-1, keepdim=True)
    return normal.permute(2, 0, 1)


def match_pcd(pc0, pc1, N=5000):
    """
    Input:
        pc0, pc1: tensor [1, N0, 3], [1, N1, 3]
        N: downsample number
    Return:
        idx_s, idx_e: [N], [N]
    """
    # Downsample with farthest point sampling
    num_fps = min(pc0.shape[1], pc1.shape[1], N)
    s_idx = farthest_point_sample(pc0, num_fps)
    pc_s = index_points(pc0, s_idx)
    e_idx = farthest_point_sample(pc1, num_fps)
    pc_e = index_points(pc1, e_idx)

    # Matching
    with torch.no_grad():
        cost = torch.cdist(pc_s, pc_e).cpu().numpy()
    idx_s, idx_e = linear_sum_assignment(cost.squeeze())
    idx_s, idx_e = s_idx[0].cpu().numpy()[idx_s], e_idx[0].cpu().numpy()[idx_e]
    return idx_s, idx_e


def match_gaussians(path, cano_gs, num_slots, visualize=False):
    print("Init canonical Gaussians by matching.")
    # load single state gaussians
    xyzs, opacities, features_dcs, features_extras, scales, rots, feats = [], [], [], [], [], [], []
    for state in (0 , 1):
        plydata = PlyData.read(path.replace('point_cloud.ply', f'point_cloud_{state}.ply'))

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])), axis=1)
        xyzs.append(xyz)
        opacities.append(np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis])

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])
        features_dcs.append(features_dc)

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (cano_gs.max_sh_degree + 1) ** 2 - 1))
        features_extras.append(features_extra)

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scale[:, idx] = np.asarray(plydata.elements[0][attr_name])
        scales.append(scale)

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rot[:, idx] = np.asarray(plydata.elements[0][attr_name])
        rots.append(rot)

        fea_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("fea")]
        feat = np.zeros((xyz.shape[0], cano_gs.fea_dim))
        for idx, attr_name in enumerate(fea_names):
            feat[:, idx] = np.asarray(plydata.elements[0][attr_name])
        feats.append(feat)

    pc0, pc1 = torch.tensor(xyzs[0])[None].cuda(), torch.tensor(xyzs[1])[None].cuda()
    idx = match_pcd(pc0, pc1) # idx: [idx_start, idx_end]

    cd, _ = chamfer_distance(pc0, pc1, batch_reduction=None, point_reduction=None) # cd: [cd_start2end, cd_end2start]
    
    larger_motion_state = 0 if cd[0].mean().item() > cd[1].mean().item() else 1
    print("Larger motion state: ", larger_motion_state)

    threshould = [cano_gs.dynamic_threshold_ratio * cd[0].max().item(), cano_gs.dynamic_threshold_ratio * cd[1].max().item()]
    mask_static = [(cd[i].squeeze() < threshould[i]).cpu().numpy() for i in range(2)]
    mask_dynamic = [~mask_static[i] for i in range(2)]

    s = larger_motion_state
    xyz = np.concatenate([xyzs[s][mask_static[s]], (xyzs[0][idx[0]] + xyzs[1][idx[1]]) * 0.5])
    opacities = np.concatenate([opacities[s][mask_static[s]], (opacities[0][idx[0]] + opacities[1][idx[1]]) * 0.5])
    features_dcs = np.concatenate([features_dcs[s][mask_static[s]], (features_dcs[0][idx[0]] + features_dcs[1][idx[1]]) * 0.5])
    features_extras = np.concatenate([features_extras[s][mask_static[s]], (features_extras[0][idx[0]] + features_extras[1][idx[1]]) * 0.5])
    scales = np.concatenate([scales[s][mask_static[s]], (scales[0][idx[0]] + scales[1][idx[1]]) * 0.5])
    rots = np.concatenate([rots[s][mask_static[s]], (rots[0][idx[0]] + rots[1][idx[1]]) * 0.5])
    feats = np.concatenate([feats[s][mask_static[s]], (feats[0][idx[0]] + feats[1][idx[1]]) * 0.5])

    cano_gs._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
    cano_gs._features_dc = nn.Parameter(torch.tensor(features_dcs, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
    cano_gs._features_rest = nn.Parameter(torch.tensor(features_extras, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
    cano_gs._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
    cano_gs._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
    cano_gs._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
    if cano_gs.fea_dim > 0:
        cano_gs.feature = nn.Parameter(torch.tensor(feats, dtype=torch.float, device="cuda").requires_grad_(True))

    cano_gs.max_radii2D = torch.zeros((cano_gs.get_xyz.shape[0]), device="cuda")
    cano_gs.active_sh_degree = cano_gs.max_sh_degree
    cano_gs.save_ply(path)

    if num_slots > 3 or 'real' in path: # larger threshold for complex or real wolrd multi-part objects
        ratio = 0.05
        threshould = [ratio * cd[0].max().item(), ratio * cd[1].max().item()]
        mask_static = [(cd[i].squeeze() < threshould[i]).cpu().numpy() for i in range(2)]
        mask_dynamic = [~mask_static[i] for i in range(2)]
    np.save(path.replace('point_cloud.ply', 'xyz_static.npy'), xyzs[s][mask_static[s]])
    np.save(path.replace('point_cloud.ply', 'xyz_dynamic.npy'), xyzs[s][mask_dynamic[s]])
    np.save(path.replace('point_cloud.ply', 'xyz_static_0.npy'), xyzs[0][mask_static[0]])
    np.save(path.replace('point_cloud.ply', 'xyz_dynamic_0.npy'), xyzs[0][mask_dynamic[0]])
    np.save(path.replace('point_cloud.ply', 'xyz_static_1.npy'), xyzs[1][mask_static[1]])
    np.save(path.replace('point_cloud.ply', 'xyz_dynamic_1.npy'), xyzs[1][mask_dynamic[1]])
    if visualize:
        import seaborn as sns
        pallete = np.array(sns.color_palette("hls", 2))
        point_cloud = o3d.geometry.PointCloud()
        x_s = xyzs[s][mask_static[s]]
        x_matched = (xyzs[0][idx[0]] + xyzs[1][idx[1]]) * 0.5
        x = np.concatenate([x_s, x_matched])
        color = np.concatenate([pallete[0].repeat(x_s.shape[0], 0), pallete[1].repeat(x_matched.shape[0], 0)])
        point_cloud.points = o3d.utility.Vector3dVector(x)
        point_cloud.colors = o3d.utility.Vector3dVector(color)
        o3d.visualization.draw_geometries([point_cloud])
    return larger_motion_state


def voxelize(xyz, voxel_size):
    """
    Voxelize the point cloud.
    Args:
        xyz: [N, 3]
        voxel_size: float
    Returns:
        voxelized_xyz: [M, 3]
    """
    xyz = np.floor(xyz / voxel_size).astype(np.int32)
    _, idx = np.unique(xyz, axis=0, return_index=True)
    return xyz[np.sort(idx)] * voxel_size


def get_refined_mask(xyzs, mask_static, mask_dynamic, voxel_size=0.05):
    new_mask_static  = []
    new_mask_dynamic = []

    for pts, m_stat, m_dyn in zip(xyzs, mask_static, mask_dynamic):
        # 원본 static/dynamic 분리
        xyz_static  = pts[m_stat]
        xyz_dynamic = pts[m_dyn]

        # static을 voxelize 한 뒤 Delaunay 삼각분할
        xyz_static_vox = voxelize(xyz_static, voxel_size)
        hull = Delaunay(xyz_static_vox)

        # dynamic 점들이 static 내부에 있으면 True
        inside = hull.find_simplex(xyz_dynamic) >= 0  # shape = (num_dynamic,)

        # 마스크 복사
        refined_stat = m_stat.copy()
        refined_dyn  = m_dyn.copy()

        # 전체 인덱스 중 dynamic으로 표시된 위치들
        dyn_idx = np.nonzero(m_dyn)[0]

        # 내부에 있는 dynamic → static으로 옮기기
        refined_stat[dyn_idx[inside]] = True
        refined_dyn [dyn_idx[inside]] = False

        new_mask_static.append(refined_stat)
        new_mask_dynamic.append(refined_dyn)

        # ── 여기서 시각화 시작 ──
        # 1) 원본 static (파랑), 원본 dynamic (빨강)
        # pcd_orig_static = o3d.geometry.PointCloud()
        # pcd_orig_static.points = o3d.utility.Vector3dVector(xyz_static)
        # pcd_orig_static.paint_uniform_color([0, 0, 1])

        # pcd_orig_dynamic = o3d.geometry.PointCloud()
        # pcd_orig_dynamic.points = o3d.utility.Vector3dVector(xyz_dynamic)
        # pcd_orig_dynamic.paint_uniform_color([1, 0, 0])

        # # 2) 리파인된 static (초록), 리파인된 dynamic (노랑)
        # xyz_refined_static  = pts[refined_stat]
        # xyz_refined_dynamic = pts[refined_dyn]

        # pcd_refined_static = o3d.geometry.PointCloud()
        # pcd_refined_static.points = o3d.utility.Vector3dVector(xyz_refined_static)
        # pcd_refined_static.paint_uniform_color([0, 1, 0])

        # pcd_refined_dynamic = o3d.geometry.PointCloud()
        # pcd_refined_dynamic.points = o3d.utility.Vector3dVector(xyz_refined_dynamic)
        # pcd_refined_dynamic.paint_uniform_color([1, 1, 0])

        # # 창 하나에 네 개의 포인트 클라우드를 동시에 표시
        # # o3d.visualization.webrtc_server.enable_webrtc()
        # o3d.visualization.draw(
        #     [
        #         pcd_orig_static,
        #         pcd_orig_dynamic,
        #         pcd_refined_static,
        #         pcd_refined_dynamic
        #     ],
        #     show_ui=True
        # )
        # ── 시각화 끝 ──

    return new_mask_static, new_mask_dynamic


def get_refined_center(xyz_static, xyz_dynamic, num_slots):
    voxel_size = 0.05
    xyz_static = voxelize(xyz_static, voxel_size)
    xyz_dynamic = voxelize(xyz_dynamic, voxel_size)

    hull = Delaunay(xyz_static)
    inside = hull.find_simplex(xyz_dynamic) >= 0
    outside_points = xyz_dynamic[~inside]

    # visualize the points
    # pcd_static = o3d.geometry.PointCloud()
    # pcd_dynamic = o3d.geometry.PointCloud()
    # pcd_static.points = o3d.utility.Vector3dVector(xyz_static)
    # pcd_dynamic.points = o3d.utility.Vector3dVector(outside_points)
    # pcd_dynamic.paint_uniform_color([1, 0, 0])  # Red for dynamic points
    # pcd_static.paint_uniform_color([0, 1, 0])  # Green for static points
    # o3d.visualization.webrtc_server.enable_webrtc()
    # o3d.visualization.draw([pcd_dynamic, pcd_static], show_ui=True)

    center = np.concatenate([xyz_static.mean(0, keepdims=True), outside_points.mean(0, keepdims=True)])
    return center


def cal_cluster_centers(cano_path, num_slots, visualize=False, fps=False):
    xyz_static = np.load(cano_path.replace('point_cloud.ply', 'xyz_static.npy'))
    xyz_dynamic = np.load(cano_path.replace('point_cloud.ply', 'xyz_dynamic.npy'))

    if fps:
        pc0, pc1 = torch.from_numpy(xyz_static).float().cuda(), torch.from_numpy(xyz_dynamic).float().cuda()
        num_points = 5000
        idx0 = farthest_point_sample(pc0[None, :, :], num_points)
        idx1 = farthest_point_sample(pc1[None, :, :], num_points)
        xyz_static = index_points(pc0[None, :, :], idx0)[0].cpu().detach().numpy()
        xyz_dynamic = index_points(pc1[None, :, :], idx1)[0].cpu().detach().numpy()
        print("Farthest point sampling done.")

    print("Finding centers by Spectral Clustering")
    if num_slots > 2:
        cluster = SpectralClustering(num_slots - 1, assign_labels='discretize', random_state=0)
        labels = cluster.fit_predict(xyz_dynamic)
        center_dynamic = np.array([xyz_dynamic[labels == i].mean(0) for i in range(num_slots - 1)])
        labels = np.concatenate([np.zeros(xyz_static.shape[0]), labels + 1])
        center = np.concatenate([xyz_static.mean(0, keepdims=True), center_dynamic])
    else:
        labels = np.concatenate([np.zeros(xyz_static.shape[0]), np.ones(xyz_dynamic.shape[0])])
        center = get_refined_center(xyz_static, xyz_dynamic, num_slots)
        # center = np.concatenate([xyz_static.mean(0, keepdims=True), xyz_dynamic.mean(0, keepdims=True)])
    x = np.concatenate([xyz_static, xyz_dynamic])
    labels = np.asarray(labels, np.int32)
    dist = (x - center[labels]) # [N, 3]
    mask = np.zeros([dist.shape[0], num_slots])
    mask[np.arange(dist.shape[0]), labels] = 1
    dist_max = (np.linalg.norm(dist, axis=-1)[:, None] * mask).max(0)[:, None] / 2 # [K, 1]
    center_info = np.concatenate([center, dist_max], -1)
    path = cano_path.replace('point_cloud.ply', 'center_info.npy')
    np.save(path, center_info)

    if visualize:
        import seaborn as sns
        pallete = np.array(sns.color_palette("hls", num_slots))
        point_cloud = o3d.geometry.PointCloud()
        c = (center[None] + np.random.randn(1000, 1, 3) * 0.05).reshape(-1, 3)
        x1 = np.concatenate([x, c], 0)
        color = np.concatenate([pallete[labels], pallete[None].repeat(1000, 0).reshape(-1, 3)], 0)
        point_cloud.points = o3d.utility.Vector3dVector(x1)
        point_cloud.colors = o3d.utility.Vector3dVector(color)

        o3d.visualization.webrtc_server.enable_webrtc()
        o3d.visualization.draw([point_cloud], show_ui=True)
import argparse
import cv2
import numpy as np
import os
import torch
import torch.nn as nn
from pathlib import Path
from plyfile import PlyData
from utils.pointnet2_utils import farthest_point_sample, index_points
import open3d as o3d
from utils.registration import registration_point_to_point_ransac, registration_point_to_plane
from utils.rotate_gaussian import transform_by_4x4, rescale_axis, translate
from scene import GaussianModel
from scipy.optimize import linear_sum_assignment
from pytorch3d.loss import chamfer_distance
import json
from utils.other_utils import cal_cluster_centers



def feature_matching(image_path1, image_path2):
    # 이미지 불러오기
    img1 = cv2.imread(image_path1)
    img2 = cv2.imread(image_path2)
    
    if img1 is None or img2 is None:
        print("Cannot read one of the images.")
        return

    # 그레이스케일로 변환 (ORB는 그레이스케일 이미지에서 동작)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # ORB 객체 생성
    orb = cv2.ORB_create()

    # 각 이미지에서 특징점과 디스크립터 검출
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    # BFMatcher 객체 생성 (ORB의 경우 Hamming distance 사용)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # 디스크립터 매칭
    matches = bf.match(des1, des2)

    # 매칭 결과를 거리(distance)가 낮은 순으로 정렬 (낮을수록 좋은 매칭)
    matches = sorted(matches, key=lambda x: x.distance)

    # 상위 50개의 매칭 결과를 선택하여 결과 이미지에 그리기
    # 상위 50개의 매칭 결과를 선택하여 결과 이미지에 그리기
    top_matches = matches[:50]
    matched_img = cv2.drawMatches(img1, kp1, img2, kp2, top_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # 매칭된 특징점의 스케일(크기) 값 출력 및 scale ratio 계산
    scale_ratios = []
    for i, m in enumerate(top_matches):
        size1 = kp1[m.queryIdx].size
        size2 = kp2[m.trainIdx].size
        ratio = size2 / size1 if size1 != 0 else 0
        scale_ratios.append(ratio)

    final_scale_avg = np.mean(scale_ratios)
    return final_scale_avg


def feature_matching_sift_ransac(img_path1, img_path2, visualize=False):
    # 이미지 로드
    img1 = cv2.imread(img_path1)
    img2 = cv2.imread(img_path2)

    if img1 is None or img2 is None:
        raise ValueError("이미지를 불러오지 못했습니다.")

    # 그레이스케일 변환
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # SIFT 생성
    sift = cv2.SIFT_create()

    # 특징점 및 디스크립터 추출
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    # BFMatcher 생성 (L2 distance 사용)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Ratio Test 적용
    good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

    if len(good_matches) < 4:
        raise RuntimeError("매칭된 특징점이 충분하지 않습니다.")

    # 매칭된 점 좌표 추출
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # RANSAC 기반 Homography 추정 (Outlier 제거)
    H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
    inlier_matches = [m for m, inl in zip(good_matches, mask.ravel()) if inl]

    top_matches = inlier_matches[:50]

    # 스케일 추정 (inlier만 사용)
    scale_ratios = []
    # for m in inlier_matches:
    for m in top_matches:
        size1 = kp1[m.queryIdx].size
        size2 = kp2[m.trainIdx].size
        if size1 > 0:
            scale_ratios.append(size2 / size1)

    avg_scale = np.mean(scale_ratios)
    inv_scale = 1.0 / avg_scale if avg_scale != 0 else 0

    print(f"[INFO] 평균 스케일 비율 (image2/image1): {avg_scale:.4f}")
    print(f"[INFO] 반전된 스케일 비율 (image1/image2): {inv_scale:.4f}")

    # 매칭 시각화
    if visualize:
        matched_img = cv2.drawMatches(img1, kp1, img2, kp2, inlier_matches, None,
                                      flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imshow("SIFT + RANSAC Matches", matched_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return avg_scale


def fps(xyz1, xyz2, npoint):
    # Downsample with farthest point sampling
    s_idx = farthest_point_sample(xyz1, npoint)
    pc_s = index_points(xyz1, s_idx)
    e_idx = farthest_point_sample(xyz2, npoint)
    pc_e = index_points(xyz2, e_idx)
    return pc_s, pc_e, s_idx, e_idx


def load_gaussians(pcd_1_path):
    gaussian = [GaussianModel(0), GaussianModel(0)]
    xyzs, nxyzs, opacities, features_dcs, features_extras, scales, rots, feats = [], [], [], [], [], [], [], []
    for state in (0, 1):
        plydata = PlyData.read(pcd_1_path.replace('gaussian_0.ply', f'gaussian_{state}.ply'))

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])), axis=1)
        xyzs.append(xyz)
        opacities.append(np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis])

        nxyz = np.stack((np.asarray(plydata.elements[0]["nx"]),
                         np.asarray(plydata.elements[0]["ny"]),
                         np.asarray(plydata.elements[0]["nz"])), axis=1)
        nxyzs.append(nxyz)

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
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (gaussian[state].max_sh_degree + 1) ** 2 - 1))
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
        feat = np.zeros((xyz.shape[0], gaussian[state].fea_dim))
        for idx, attr_name in enumerate(fea_names):
            feat[:, idx] = np.asarray(plydata.elements[0][attr_name])
        feats.append(feat)

        gaussian[state]._xyz = nn.Parameter(torch.from_numpy(xyz).float().cuda(), requires_grad=True)
        gaussian[state]._features_dc = nn.Parameter(torch.from_numpy(features_dc).float().cuda(), requires_grad=True)
        gaussian[state]._features_rest = nn.Parameter(torch.from_numpy(features_extra).float().cuda(), requires_grad=True)
        gaussian[state]._opacity = nn.Parameter(torch.from_numpy(opacities[state]).float().cuda(), requires_grad=True)
        gaussian[state]._scaling = nn.Parameter(torch.from_numpy(scale).float().cuda(), requires_grad=True)
        gaussian[state]._rotation = nn.Parameter(torch.from_numpy(rot).float().cuda(), requires_grad=True)
        gaussian[state].max_radii2D = torch.zeros((gaussian[state].get_xyz.shape[0]), device="cuda")
        gaussian[state].active_sh_degree = gaussian[state].max_sh_degree

    return gaussian, xyzs, nxyzs, opacities, features_dcs, features_extras, scales, rots, feats


def match_pcd(pc_s, s_idx, pc_e, e_idx):
    # Matching
    with torch.no_grad():
        cost = torch.cdist(pc_s, pc_e).cpu().numpy()
    idx_s, idx_e = linear_sum_assignment(cost.squeeze())
    idx_s, idx_e = s_idx[0].cpu().numpy()[idx_s], e_idx[0].cpu().numpy()[idx_e]
    return idx_s, idx_e


def get_cano_gs(gaussian, xyzs, opacities, features_dcs, features_extras, scales, rots, feats, idx, mask_static, s):

    # visualize each xyz using open3d
    # pcd1 = o3d.geometry.PointCloud()
    # pcd1.points = o3d.utility.Vector3dVector(xyzs[0][idx[0]])
    # pcd1.paint_uniform_color([1, 0, 0])  # Red color

    # pcd2 = o3d.geometry.PointCloud()
    # pcd2.points = o3d.utility.Vector3dVector(xyzs[1][idx[1]])
    # pcd2.paint_uniform_color([0, 1, 0])  # Green color

    # o3d.visualization.webrtc_server.enable_webrtc()
    # o3d.visualization.draw([pcd1, pcd2], show_ui=True)


    # xyz = (xyzs[0][idx[0]] + xyzs[1][idx[1]]) * 0.5
    # opacities = (opacities[0][idx[0]] + opacities[1][idx[1]]) * 0.5
    # features_dcs = (features_dcs[0][idx[0]] + features_dcs[1][idx[1]]) * 0.5
    # features_extras = (features_extras[0][idx[0]] + features_extras[1][idx[1]]) * 0.5
    # scales = (scales[0][idx[0]] + scales[1][idx[1]]) * 0.5
    # rots = (rots[0][idx[0]] + rots[1][idx[1]]) * 0.5

    static_xyz = torch.tensor(xyzs[s][mask_static[s]]).unsqueeze(0).cuda()
    static_fps_idx = farthest_point_sample(static_xyz, 20000).cpu().detach().numpy()[0]

    # xyz = np.concatenate([xyzs[s][mask_static[s]], (xyzs[0][idx[0]] + xyzs[1][idx[1]]) * 0.5])
    # opacities = np.concatenate([opacities[s][mask_static[s]], (opacities[0][idx[0]] + opacities[1][idx[1]]) * 0.5])
    # features_dcs = np.concatenate([features_dcs[s][mask_static[s]], (features_dcs[0][idx[0]] + features_dcs[1][idx[1]]) * 0.5])
    # features_extras = np.concatenate([features_extras[s][mask_static[s]], (features_extras[0][idx[0]] + features_extras[1][idx[1]]) * 0.5])
    # scales = np.concatenate([scales[s][mask_static[s]], (scales[0][idx[0]] + scales[1][idx[1]]) * 0.5])
    # rots = np.concatenate([rots[s][mask_static[s]], (rots[0][idx[0]] + rots[1][idx[1]]) * 0.5])

    xyz = np.concatenate([xyzs[s][mask_static[s]][static_fps_idx], (xyzs[0][idx[0]] + xyzs[1][idx[1]]) * 0.5])
    opacities = np.concatenate([opacities[s][mask_static[s]][static_fps_idx], (opacities[0][idx[0]] + opacities[1][idx[1]]) * 0.5])
    features_dcs = np.concatenate([features_dcs[s][mask_static[s]][static_fps_idx], (features_dcs[0][idx[0]] + features_dcs[1][idx[1]]) * 0.5])
    features_extras = np.concatenate([features_extras[s][mask_static[s]][static_fps_idx], (features_extras[0][idx[0]] + features_extras[1][idx[1]]) * 0.5])
    scales = np.concatenate([scales[s][mask_static[s]][static_fps_idx], (scales[0][idx[0]] + scales[1][idx[1]]) * 0.5]) + np.log(4.0)
    rots = np.concatenate([rots[s][mask_static[s]][static_fps_idx], (rots[0][idx[0]] + rots[1][idx[1]]) * 0.5])
    
    # feats = np.concatenate([feats[s][mask_static[s]], (feats[0][idx[0]] + feats[1][idx[1]]) * 0.5])
    
    cano_gs = GaussianModel(0)
    cano_gs._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
    cano_gs._features_dc = nn.Parameter(torch.tensor(features_dcs, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
    cano_gs._features_rest = nn.Parameter(torch.tensor(features_extras, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
    cano_gs._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
    cano_gs._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
    cano_gs._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
    if cano_gs.fea_dim > 0:
        feats = (feats[0][idx[0]] + feats[1][idx[1]]) * 0.5
        cano_gs.feature = nn.Parameter(torch.tensor(feats, dtype=torch.float, device="cuda").requires_grad_(True))

    cano_gs.max_radii2D = torch.zeros((cano_gs.get_xyz.shape[0]), device="cuda")
    cano_gs.active_sh_degree = cano_gs.max_sh_degree
    return cano_gs


def main(image1_path, image2_path, pcd1_path, pcd2_path, num_slots):
    '''
        image feature matching -> scaling ->ICP registration (get transform matrix) -> Gaussian rotation ->
        view rendering -> FPS -> hungarian matching -> get cano_gs/static_gs/dynamic_gs/larger_state/center_info
    '''
    # scale_factor = feature_matching(image1_path, image2_path)
    # scale_factor = feature_matching_sift_ransac(image1_path, image2_path)

    image1_path = str(image1_path)
    image2_path = str(image2_path)
    pcd1_path = str(pcd1_path)
    pcd2_path = str(pcd2_path)

    # Load the Gaussian models
    gs, xyz, nxyzs, opacities, features_dcs, features_extras, scales, rots, feats = load_gaussians(pcd1_path)

    
    scale_factor_path = pcd1_path.replace('gaussian_0.ply', 'scaling.npy')
    if os.path.exists(scale_factor_path):
        scale_factor = np.load(scale_factor_path)
    else:
        raise FileNotFoundError(f"Scaling factor file not found: {scale_factor_path}")

    # apply scaling and transformation to the second Gaussian
    gs[1] = rescale_axis(gs[1], scale_factor)


    translate_factor_path = pcd1_path.replace('gaussian_0.ply', 'translation.npy')
    if os.path.exists(translate_factor_path):
        translate_factor = np.load(translate_factor_path)
    else:
        raise FileNotFoundError(f"Translation factor file not found: {translate_factor_path}")
    gs[1] = translate(gs[1], translate_factor)

    # Save the transformed Gaussian
    output_path = pcd2_path.replace(".ply", "_transformed.ply")
    gs[1].save_ply(output_path)
    print(f"Transformed Gaussian saved to {output_path}")

    # swap the second gaussian properties
    xyz[1] = gs[1]._xyz.cpu().detach().numpy()
    opacities[1] = gs[1]._opacity.cpu().detach().numpy()
    features_dcs[1] = gs[1]._features_dc.cpu().detach().numpy()
    features_extras[1] = gs[1]._features_rest.cpu().detach().numpy()
    scales[1] = gs[1]._scaling.cpu().detach().numpy()
    rots[1] = gs[1]._rotation.cpu().detach().numpy()
    feats[1] = gs[1].feature.cpu().detach().numpy()

    # FPS
    num_fps = 5000
    xyz_1 = gs[0]._xyz.unsqueeze(0)
    xyz_2 = gs[1]._xyz.unsqueeze(0)
    xyz_1_fps, xyz2_new_fps, idx1, xyz2_new_fps_idx = fps(xyz_1, xyz_2, num_fps)


    # # save the matched point clouds
    xyz_1_fps_pcd = o3d.geometry.PointCloud()
    xyz_1_fps_pcd.points = o3d.utility.Vector3dVector(xyz_1_fps[0].cpu().detach().numpy())
    xyz_1_fps_pcd.paint_uniform_color([1, 0, 0])  # Red color

    xyz2_new_fps_pcd = o3d.geometry.PointCloud()
    xyz2_new_fps_pcd.points = o3d.utility.Vector3dVector(xyz2_new_fps[0].cpu().detach().numpy())
    xyz2_new_fps_pcd.paint_uniform_color([0, 0, 1])  # Blue color

    # Save the point clouds to a file
    o3d.io.write_point_cloud(pcd1_path.replace("gaussian_0.ply", "matched_pcd.ply"), xyz_1_fps_pcd + xyz2_new_fps_pcd)
    print(f"Matched point clouds saved to {pcd1_path.replace('gaussian_0.ply', 'matched_pcd.ply')}")

    # # Match the FPS points
    idx_s, idx_e = match_pcd(xyz_1_fps, idx1, xyz2_new_fps, xyz2_new_fps_idx)

    cd, _ = chamfer_distance(xyz_1, xyz_2, batch_reduction=None, point_reduction=None) # cd: [cd_start2end, cd_end2start]
    larger_motion_state = 0 if cd[0].mean().item() > cd[1].mean().item() else 1
    s = larger_motion_state
    print("Larger motion state: ", larger_motion_state)

    threshould = [0.05 * cd[0].max().item(), 0.05 * cd[1].max().item()]
    mask_static = [(cd[i].squeeze() < threshould[i]).cpu().numpy() for i in range(2)]
    mask_dynamic = [~mask_static[i] for i in range(2)]

    # # get cano_gs
    cano_gs = get_cano_gs(gs, xyz, opacities, features_dcs, features_extras, scales, rots, feats, (idx_s, idx_e), mask_static, s)
    cano_gs.save_ply(pcd1_path.replace("gaussian_0.ply", "point_cloud.ply"))
    print(f"Canonical Gaussian saved to {pcd1_path.replace('gaussian_0.ply', 'point_cloud.ply')}")

    
    if num_slots > 3 or 'real' in pcd1_path: # larger threshold for complex or real wolrd multi-part objects
        ratio = 0.15
        threshould = [ratio * cd[0].max().item(), ratio * cd[1].max().item()]
        mask_static = [(cd[i].squeeze() < threshould[i]).cpu().numpy() for i in range(2)]
        mask_dynamic = [~mask_static[i] for i in range(2)]        

    np.save(pcd1_path.replace('gaussian_0.ply', 'xyz_static.npy'), xyz[s][mask_static[s]])
    np.save(pcd1_path.replace('gaussian_0.ply', 'xyz_dynamic.npy'), xyz[s][mask_dynamic[s]])
    np.save(pcd1_path.replace('gaussian_0.ply', 'xyz_static_0.npy'), xyz[0][mask_static[0]])
    np.save(pcd1_path.replace('gaussian_0.ply', 'xyz_dynamic_0.npy'), xyz[0][mask_dynamic[0]])
    np.save(pcd1_path.replace('gaussian_0.ply', 'xyz_static_1.npy'), xyz[1][mask_static[1]])
    np.save(pcd1_path.replace('gaussian_0.ply', 'xyz_dynamic_1.npy'), xyz[1][mask_dynamic[1]])
    
    # save static mask
    np.save(pcd1_path.replace('gaussian_0.ply', 'mask_static.npy'), mask_static[s])
    print("Static and dynamic points saved.")

    cal_cluster_centers(pcd1_path.replace('gaussian_0.ply', 'point_cloud.ply'), num_slots, visualize=False, fps=True)
    print("Cluster centers saved.")

    # o3d.visualization.webrtc_server.enable_webrtc()
    # o3d.visualization.draw([point_cloud, point_cloud2], show_ui=True)

    return larger_motion_state


def _default_paths(dataset: str, subset: str, scene: str, coarse_name: str, iteration: int):
    coarse_root = Path("outputs") / dataset / subset / scene / coarse_name / "point_cloud" / f"iteration_{iteration}"
    images_root = Path("data") / dataset / subset / scene

    def _first_png(p: Path):
        candidates = sorted(p.glob("*.png"))
        return candidates[0] if candidates else None

    # Prefer start/end RGBA frames if they exist; fall back to legacy input_0/input_1 names.
    img0 = _first_png(images_root / "start" / "train" / "rgba")
    img1 = _first_png(images_root / "end" / "train" / "rgba")
    if img0 is None:
        legacy0 = images_root / "input_0.png"
        img0 = legacy0 if legacy0.exists() else None
    if img1 is None:
        legacy1 = images_root / "input_1.png"
        img1 = legacy1 if legacy1.exists() else None

    if img0 is None or img1 is None:
        raise FileNotFoundError(
            f"Could not find input images under {images_root}/start/train/rgba or end/train/rgba "
            f"(also checked input_0.png/input_1.png)."
        )

    return (img0, img1, coarse_root / "gaussian_0.ply", coarse_root / "gaussian_1.ply")


def _parse_args():
    parser = argparse.ArgumentParser(description="Align two coarse Gaussians and generate canonical assets.")
    parser.add_argument("--dataset", default="artgs", help="Dataset name used to build default paths.")
    parser.add_argument("--subset", default="sapien", help="Subset name used to build default paths.")
    parser.add_argument("--scene_name", required=True, help="Scene name used to build default paths.")
    parser.add_argument("--coarse_name", default="coarse_gs", help="Coarse model directory name.")
    parser.add_argument("--iteration", type=int, default=10000, help="Iteration folder to read coarse Gaussians from.")
    parser.add_argument("--image0", type=Path, help="Path to input_0.png. Overrides dataset/subset/scene defaults.")
    parser.add_argument("--image1", type=Path, help="Path to input_1.png. Overrides dataset/subset/scene defaults.")
    parser.add_argument("--pcd0", type=Path, help="Path to gaussian_0.ply. Overrides dataset/subset/scene defaults.")
    parser.add_argument("--pcd1", type=Path, help="Path to gaussian_1.ply. Overrides dataset/subset/scene defaults.")
    parser.add_argument("--num_slots", type=int, help="Number of slots. Defaults to arguments/num_slots.json.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    image1_path, image2_path, pcd1_path, pcd2_path = args.image0, args.image1, args.pcd0, args.pcd1

    if any(p is None for p in (image1_path, image2_path, pcd1_path, pcd2_path)):
        image1_path, image2_path, pcd1_path, pcd2_path = _default_paths(
            args.dataset, args.subset, args.scene_name, args.coarse_name, args.iteration
        )

    num_slots = args.num_slots
    if num_slots is None:
        slots = json.load(open("./arguments/num_slots.json", "r"))
        num_slots = slots[args.dataset][args.subset][args.scene_name]

    for path in (image1_path, image2_path, pcd1_path, pcd2_path):
        if not Path(path).exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    larger_motion_state = main(image1_path, image2_path, pcd1_path, pcd2_path, num_slots)
    file = json.load(open("arguments/larger_motion_state.json", "r"))
    file[args.dataset][args.subset][args.scene_name] = larger_motion_state
    json.dump(file, open("arguments/larger_motion_state.json", "w"), indent=4)

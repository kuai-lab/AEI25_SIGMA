import numpy as np
import torch
import torch.nn as nn

def rescale(gaussian, scale_factor):
    if scale_factor != 1.:
        gaussian._xyz = gaussian._xyz * scale_factor
        gaussian._scaling = gaussian._scaling + np.log(scale_factor)
        print("rescaled with factor {}".format(scale_factor))

    return gaussian

def rescale_axis(gaussian, scale_factors):
    """
    Gaussian의 xyz를 축별로 비등방(비균일) 스케일링합니다.

    Args:
        gaussian: 변환할 Gaussian 객체 (._xyz, ._scaling 속성 보유).
        scale_factors: 길이 3의 iterable, 각 축(x, y, z)에 대한 스케일링 계수.

    Returns:
        스케일링이 적용된 Gaussian 객체.
    """
    s = np.array(scale_factors, dtype=np.float32)
    if not np.allclose(s, 1.0):
        # 1) xyz 좌표 업데이트
        xyz = gaussian._xyz.cpu().detach().numpy()  # (N,3)
        xyz = xyz * s[np.newaxis, :]
        gaussian._xyz = nn.Parameter(
            torch.from_numpy(xyz).float().to(gaussian._xyz.device),
            requires_grad=True
        )

        # 2) 전체 부피 스케일 로그로 업데이트 (optional)
        #    uniform 스케일의 경우 np.log(scale_factor) 와 동일한 개념으로,
        #    비균일 스케일의 부피 변화는 det(diag(s)) = prod(s) 이므로
        #    스케일 로그는 sum(log(s))
        gaussian._scaling = gaussian._scaling + np.sum(np.log(s))

        print(f"rescaled axes with factors {scale_factors}")

    return gaussian


def translate(gaussian, translation):
    """
    Gaussian의 xyz 좌표에만 단순 이동 변환을 적용합니다.

    Args:
        gaussian: 변환할 Gaussian 객체 (._xyz 속성 보유).
        translation: 길이 3의 iterable, (dx, dy, dz).

    Returns:
        이동이 적용된 Gaussian 객체.
    """
    t = np.array(translation, dtype=np.float32)
    # 1) xyz 좌표 업데이트
    xyz = gaussian._xyz.cpu().detach().numpy()  # (N,3)
    xyz = xyz + t[np.newaxis, :]
    gaussian._xyz = nn.Parameter(
        torch.from_numpy(xyz).float().to(gaussian._xyz.device),
        requires_grad=True
    )

    print(f"translated by {translation}")
    return gaussian


def quat_multiply(quaternion0, quaternion1):
    """
    두 quaternion (shape: (..., 4)) 의 곱셈을 수행합니다.
    convention: quaternion은 (x, y, z, w) 순서라고 가정합니다.
    """
    w0, x0, y0, z0 = np.split(quaternion0, 4, axis=-1)
    w1, x1, y1, z1 = np.split(quaternion1, 4, axis=-1)
    return np.concatenate((
        -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
            x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
        -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
            x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0,
    ), axis=-1)

def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec

def transform_by_4x4(gaussian, T: np.ndarray, keep_sh_degree: bool = True):
    """
    4x4 변환행렬 T를 사용해 Gaussian을 변환합니다.
        - T[:3, :3] 은 회전 행렬 (R)
        - T[:3, 3] 은 평행이동 벡터 (t)
    """
    # 1. xyz 좌표에 회전 후 평행이동 적용
    R = T[:3, :3]
    t = T[:3, 3]
    xyz = np.asarray(gaussian._xyz.cpu().detach().numpy())
    # gaussian._xyz = nn.Parameter(torch.tensor(np.matmul(xyz, R.T))) + t
    xyz = np.matmul(xyz, R.T) + t
    gaussian._xyz = nn.Parameter(torch.from_numpy(xyz).float().cuda(), requires_grad=True)

    # 2. quaternion (회전 정보) 변환: 회전행렬 R을 quaternion으로 변환 후 곱셈 적용
    quaternions = rotmat2qvec(R)[np.newaxis, ...]
    rotation = np.asarray(gaussian._rotation.cpu().detach().numpy())
    rotation = quat_multiply(rotation, quaternions)
    rotation = rotation / np.linalg.norm(rotation, axis=-1, keepdims=True)

    gaussian._rotation = nn.Parameter(torch.from_numpy(rotation).float().cuda(), requires_grad=True)

    # 3. TODO: SH 계수 회전 (필요하다면 구현). 여기서는 옵션에 따라 SH 차수를 0으로 설정
    if not keep_sh_degree:
        print("set sh_degree=0 when 4x4 transformation is applied")
        gaussian._sh_degrees = 0

    return gaussian

    


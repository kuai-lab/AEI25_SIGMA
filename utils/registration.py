import open3d as o3d
import numpy as np
import copy

def preprocess_for_ransac(pcd, voxel_size):
    """
    RANSAC registration을 위한 다운샘플링 및 FPFH 특징 추출.
    - pcd: 원본 포인트 클라우드
    - voxel_size: 복셀 크기
    """
    # print("RANSAC: Voxel downsampling with voxel size:", voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)
    
    # print("RANSAC: Estimating normals...")
    pcd_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    
    # print("RANSAC: Computing FPFH features...")
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=100))
    
    return pcd_down, fpfh

def registration_point_to_point_ransac(source, target, voxel_size, max_iterations=400000, max_validation=500):
    """
    Coarse alignment을 위해 RANSAC 기반 point-to-point registration 수행.
    FPFH 특징을 기반으로 correspondence를 찾고, TransformationEstimationPointToPoint()로 초기 변환을 추정합니다.
    """
    # print("Preprocessing source and target for RANSAC-based registration...")
    source_down, source_fpfh = preprocess_for_ransac(source, voxel_size)
    target_down, target_fpfh = preprocess_for_ransac(target, voxel_size)
    
    # RANSAC에서 사용할 correspondence 탐색 범위 (거리 임계값)
    distance_threshold = voxel_size * 3
    # print("RANSAC: Using distance threshold:", distance_threshold)
    
    # print("Performing RANSAC based point-to-point registration...")
    result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down,
        source_fpfh, target_fpfh, True,  # mutual filter 활성화
        distance_threshold,
        # Point-to-point estimation (scale은 보통 사용하지 않음)
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        4,  # correspondence set의 최소 포인트 개수
        [o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
        o3d.pipelines.registration.RANSACConvergenceCriteria(max_iterations, max_validation)
    )
    
    return result_ransac

def preprocess_for_p2l(pcd, voxel_size):
    """
    Point-to-plane registration을 위한 다운샘플링 및 노멀 추정.
    point-to-plane ICP는 노멀 정보가 반드시 필요하므로 별도로 수행합니다.
    """
    # print("P2L: Voxel downsampling with voxel size:", voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)
    
    # print("P2L: Estimating normals...")
    pcd_down.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2, max_nn=30))
    
    return pcd_down

def registration_point_to_plane(source, target, init_transformation, voxel_size, max_iterations=50, tolerance=1e-6):
    """
    Refinement 단계: 초기 변환(예: RANSAC 결과)을 바탕으로 point-to-plane ICP 실행.
    다운샘플된 데이터를 사용하며, 작은 임계값 내에서 미세 조정을 수행합니다.
    """
    # print("P2L: Preprocessing source and target for point-to-plane refinement...")
    source_down = preprocess_for_p2l(source, voxel_size)
    target_down = preprocess_for_p2l(target, voxel_size)
    
    # 미세 조정을 위한 threshold (더 작게 설정)
    threshold = voxel_size * 1.0    # originally 0.5
    # print("P2L: Using distance threshold for ICP refinement:", threshold)
    
    # print("Performing point-to-plane ICP refinement...")
    result_p2l = o3d.pipelines.registration.registration_icp(
        source_down, target_down, threshold, init_transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iterations, relative_fitness=tolerance)
    )
    
    return result_p2l




if __name__ == "__main__":
    # 포인트 클라우드 파일 경로 설정
    source_path = "plys/trashcan_0.ply"
    target_path = "plys/scaled_trashcan_1.ply"
    
    # 포인트 클라우드 불러오기
    source = o3d.io.read_point_cloud(source_path)
    target = o3d.io.read_point_cloud(target_path)
    
    # 원본 고해상도 소스 클라우드를 복사 (refinement 후 시각화용)
    source_orig = copy.deepcopy(source)
    
    # 복셀 다운샘플링에 사용할 voxel 크기 설정
    voxel_size = 0.05
    
    # 1. Coarse alignment: RANSAC 기반의 point-to-point registration (특징을 이용)
    result_ransac = registration_point_to_point_ransac(source, target, voxel_size,
                                                       max_iterations=400000, max_validation=500)
    # print("RANSAC-based Point-to-Point ICP Transformation:")
    # print(result_ransac.transformation)
    # print("RANSAC-based Inlier RMSE:", result_ransac.inlier_rmse)
    
    # 2. Refinement: Point-to-plane ICP (초기값은 RANSAC 결과 사용)
    result_p2l = registration_point_to_plane(source, target, result_ransac.transformation,
                                               voxel_size, max_iterations=50, tolerance=1e-6)
    # print("Refinement (Point-to-Plane ICP) Transformation:")
    # print(result_p2l.transformation)
    # print("Refinement Inlier RMSE:", result_p2l.inlier_rmse)
    
    # 최종 transformation을 고해상도 source 클라우드에 적용
    source_orig.transform(result_p2l.transformation)
    
    # 색상을 지정하여 결과를 시각적으로 구분
    source_orig.paint_uniform_color([0, 0.651, 0.929])  # 변환된 소스: 파란색 계열
    target.paint_uniform_color([1, 0.706, 0])            # 타겟: 노란색 계열
    
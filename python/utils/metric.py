import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial import KDTree

def compute_chamfer_distance(pcd1, pcd2):
    """
    Args:
        pcd1 : o3d.geometry.PointCloud
        pcd2 : o3d.geometry.PointCloud
    Return:
        chamfer_distance : float
    """
    
    distance_1_to_2 = np.asarray(pcd1.compute_point_cloud_distance(pcd2))
    distance_2_to_1 = np.asarray(pcd2.compute_point_cloud_distance(pcd1))
    
    return distance_1_to_2.mean() + distance_2_to_1.mean()

# def compute_surface_coverage(gt_cloud, compare_cloud, threshold=0.005):
#     """
#     Args:
#         gt_cloud : o3d.geometry.PointCloud
#         compare_cloud : o3d.geometry.PointCloud
#         threshold : float
#     Return:
#         surface_coverage : float
#     """
    
#     gt_points = np.asarray(gt_cloud.points)

#     compare_points = np.asarray(compare_cloud.points)

#     gt_tree = cKDTree(gt_points)
    
#     distance, indices = gt_tree.query(compare_points, distance_upper_bound=threshold)

#     valid_indices = indices < len(gt_points)
    
#     unique_assigned_gt_indices = set(indices[valid_indices])
    
#     return len(unique_assigned_gt_indices) / len(gt_points)
 
def compute_surface_coverage(gt_cloud, compare_cloud, threshold=0.005):
    """
    Args:
        gt_cloud : o3d.geometry.PointCloud
        compare_cloud : o3d.geometry.PointCloud
        threshold : float
    Return:
        surface_coverage : float
    """
    
    gt_points = np.asarray(gt_cloud.points)

    compare_points = np.asarray(compare_cloud.points)

    # compare_tree = cKDTree(compare_points)
    compare_tree = KDTree(compare_points)
    
    distance, indices = compare_tree.query(gt_points, distance_upper_bound=threshold)

    covered_points = np.sum(distance <= threshold)
    
    coverage = (covered_points / len(gt_points))
    print(f"Threshold : {threshold}")
    print(f"Compare points : {len(compare_points)}")
    print(f"Covered points : {covered_points}, GT points : {len(gt_points)}, coverage : {coverage}")
    return coverage
    
def look_at(center, eye, up):
    
    f = np.array(center) - np.array(eye)
    f = f / np.linalg.norm(f)
    u = np.array(up)
    u = u / np.linalg.norm(u)
    s = np.cross(f, u)
    s = s / np.linalg.norm(s)
    u = np.cross(s, f)
    
    M = np.identity(4)
    M[:3, :3] = np.vstack([s, u, -f])
    T = np.identity(4)
    T[:3, 3] = -np.array(eye)
    return M @ T
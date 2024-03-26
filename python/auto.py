import numpy as np
import cv2
import open3d as o3d
import glob
import os

from .config import *
from .VoxelGrid import PyVoxelGrid
from .utils.metric import look_at, compute_chamfer_distance, compute_surface_coverage

DATAS = ["color", "pose", "surface_coverage", "entropy", "policy_image"]

INITIAL_IMAEG_NUM = 155

def run(args):
    
    ## Read arguments
    datapath = args.data
    policy = args.policy
    modelpath = args.model
    show = args.show
    
    ## Make save directories
    modelname = os.path.split(modelpath)[-1].split(".")[0]
    savepath = os.path.join(SAVE_ROOT_PATH, modelname)
    os.makedirs(savepath, exist_ok=True)
    n_dirs_policy = len([name for name in os.listdir(savepath) if os.path.isdir(os.path.join(savepath, name)) and name.startswith(f"policy{policy}")])
    savepath = os.path.join(savepath, f"policy{policy}_{n_dirs_policy}")
    os.makedirs(savepath, exist_ok=True)
    for DATA in DATAS:
        os.makedirs(os.path.join(savepath, DATA), exist_ok=True)
    
    ## Save list
    surface_coverages = []
    selected_images = []
    selected_poses = []
    
    ## Print information
    print(f"Model : {modelname}")
    print(f"Policy : {policy}") 
    
    ## set dataset path
    color_image_path_list = glob.glob(os.path.join(datapath, "color", "*.png"))
    depth_image_path_list = glob.glob(os.path.join(datapath, "depth", "*.npy"))
    pose_path_list = glob.glob(os.path.join(datapath, "pose", "*.txt"))
    intrinsic_path_list = glob.glob(os.path.join(datapath, "intrinsic", "*.txt"))
    pointcloud_path_list = glob.glob(os.path.join(datapath, "pointcloud", "*.npy"))
    print(f"Number of images : color {len(color_image_path_list)}, depth {len(depth_image_path_list)}")
 
    ## Load model   
    model = o3d.io.read_triangle_mesh(modelpath)
    model = fit_model_aabb_size(model, MODEL_SIZE)
 
    ## Groundtruth pointcloud   
    model_gt_pcd = model.sample_points_uniformly(NUMBER_MODEL_SAMPLE_POINTS)
    
    ## VoxelGrid initilaize
    voxelgrid = PyVoxelGrid(VOXEL_BNDS, VOXEL_SIZE)
    
    close = False
    
    SELECT = INITIAL_IMAEG_NUM
    pcd_compare = o3d.geometry.PointCloud()
    for n in range(MAX_ITERATE):
        
        color = cv2.imread(color_image_path_list[SELECT])
        depth = np.load(depth_image_path_list[SELECT])
        pose = np.loadtxt(pose_path_list[SELECT])
        intrinsic = np.loadtxt(intrinsic_path_list[SELECT])
        pointcloud = np.load(pointcloud_path_list[SELECT])
        
        image, value = voxelgrid.get_policies(policy, color.shape[0], color.shape[1],
                                                  intrinsic, np.linalg.inv(pose))
        
        voxelgrid.integrate(color.shape[0], color.shape[1],
                            color, depth, intrinsic, np.linalg.inv(pose))
        
        color_image_path_list.pop(SELECT)
        depth_image_path_list.pop(SELECT)
        pose_path_list.pop(SELECT)
        intrinsic_path_list.pop(SELECT)
        pointcloud_path_list.pop(SELECT)
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pointcloud)
        pcd_compare += pcd
        
        surface_coverage = compute_surface_coverage(model_gt_pcd, pcd_compare, threshold=SURFACE_THRESHOLD)
        surface_coverages.append(surface_coverage)
        
        cv2.imwrite(os.path.join(savepath, "color", "%d.png" % n), color)
        np.savetxt(os.path.join(savepath, "pose", "%d.txt" % n), pose)
        cv2.imwrite(os.path.join(savepath, "policy_image", "%d.png" % n), image)


        max_index = None
        max_value = None
        for i, (pose_path, intrinsic_path) in enumerate(zip(pose_path_list, intrinsic_path_list)):
        
            pose = np.loadtxt(pose_path)
            intrinsic = np.loadtxt(intrinsic_path)
            
            image, value = voxelgrid.get_policies(policy, color.shape[0], color.shape[1],
                                                  intrinsic, np.linalg.inv(pose))
            print(f"POLICY : [{policy}], View : [{n+1}/{MAX_ITERATE}], sc : {surface_coverage:.5f}, Test : [{i+1}/{len(pose_path_list)}] value : {value:.5f}")
            if max_value is None:
                max_value = value
                max_index = i
            elif max_value < value:
                max_value = value
                max_index = i    
                            
            if show:
                cv2.imshow("image", image)
                key = cv2.waitKey(1)
                if key == ord('q'):
                    close = True
                    break
        if close:
            break
        
        SELECT = max_index
    
    np.savetxt(os.path.join(savepath, "surface_coverage", "surface_coverage.txt"), np.asarray(surface_coverages))        
        
    
    
    
    
    
def fit_model_aabb_size(model, aabb_size):
    
    model_bb = model.get_axis_aligned_bounding_box()
    model = model.scale(aabb_size / (model_bb.get_max_bound() - model_bb.get_min_bound()).max(), [0, 0, 0])
    model_bb = model.get_axis_aligned_bounding_box()
    model = model.translate(-model_bb.get_center())
    
    return model
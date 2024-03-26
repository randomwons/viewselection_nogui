from .config import *

from DirectionVoxelGrid import VoxelGrid

import numpy as np
import open3d as o3d
import cv2
import matplotlib.pyplot as plt

class PyVoxelGrid(VoxelGrid):
    def __init__(self, voxel_bnds, voxel_size):
        voxel_bnds = np.array(voxel_bnds)
        super().__init__(voxel_bnds, voxel_size)
    
    def get_policies(self, policy, height, width, intrinsic, pose):
        
        if policy == 0: image = super().occlusion_aware(height, width, intrinsic, pose)
        elif policy == 1: image = super().unobserved_voxel(height, width, intrinsic, pose)
        elif policy == 2: image = super().rear_side_voxel(height, width, intrinsic, pose)
        elif policy == 3: image = super().rear_side_entropy(height, width, intrinsic, pose)
        elif policy == 4: image = super().krigel(height, width, intrinsic, pose)
        elif policy == 5: image = super().occlusion_aware_face(height, width, intrinsic, pose)
        elif policy == 6: image = super().unobserved_voxel_face(height, width, intrinsic, pose)
        elif policy == 7: image = super().rear_side_voxel_face(height, width, intrinsic, pose)
        elif policy == 8: image = super().rear_side_entropy_face(height, width, intrinsic, pose)
        elif policy == 9: image = super().krigel_face(height, width, intrinsic, pose)
        
        value = image.sum()
        
        # image = normazlie_and_color(image)
        return normazlie_and_color(image), value
        
        
def normazlie_and_color(image):
    
    mask = (image == 0)
    image_nonzero = image[~mask]
    if image_nonzero.shape[0] == 0:
        return np.ones((image.shape[0], image.shape[1], 3), dtype=np.uint8) * 255
    
    normalized = cv2.normalize(image_nonzero, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    normalized = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
    normalized_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    normalized_image[~mask] = normalized[:, 0, :]
    normalized_image[mask] = 255
    return normalized_image
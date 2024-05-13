

from competition_pipeline.utils import segment_cloth, filter_pc, plot_pc
import cv2
import numpy as np
import time
from sklearn.neighbors import NearestNeighbors
import os

def sample_grasp(camera_pose_in_world,
                 camera_intrinsics,
                 camera_resolution,
                 image_rgb,
                 point_cloud):
    '''Give the observations required, return the grasp pose'''
    # Segment the cloth first
    mask = segment_cloth(image_rgb, camera_pose_in_world, camera_intrinsics, camera_resolution)
    # mask = cv2.imread('../log/mask_comp.png', cv2.IMREAD_GRAYSCALE)
    # current_dir = os.path.dirname(__file__)
    # log_dir = os.path.join(current_dir, "log")
    # mask = cv2.imread(os.path.join(log_dir, "mask.png"))[:, :, 0]
    print("Masking finished")

    # Get the point cloud
    pc = filter_pc(point_cloud, mask, camera_resolution)
    
    z_offset = 0.3
    # sr_Y_min, sr_Y_max = self.Y_range
    # set region of interest (roi)
    pc = pc[pc[:, 0] < 0.2]
    x = pc[:, 0]
    y = pc[:, 1]
    z = pc[:, 2]
    pc_origin = pc.copy()
    
    z_min, z_max = np.min(z), np.max(z)
    if z_max - z_min <= z_offset:
        # if z_max - z_min is too small, set z_offset to 0.5*diff
        z_offset = (z_max - z_min) * 0.5
    z_max = z_max - z_offset # manually set max height for grasping

    # Only retain those points whose z is lower or equal to z_max
    pc = pc[pc[:, 2] <= z_max]

    # sample a point with z in [z_roi_min, z_roi_max] and y in [sr_Y_min, sr_Y_max]
    # idx_arr = np.where((z >= z_roi_min) & (z <= z_roi_max) & (y >= sr_Y_min) & (y <= sr_Y_max))[0]
    idx_arr = list(range(0, pc.shape[0]))
    if len(idx_arr) > 0:

        point_candidates = pc[idx_arr]
        nbrs = NearestNeighbors(n_neighbors=4, algorithm='auto').fit(point_candidates)
        distances, indices = nbrs.kneighbors(point_candidates)

        local_maxima = []

        for i in range(point_candidates.shape[0]):
            x_center = point_candidates[i, 0]
            neighbor_indices = indices[i, 1:]  # 排除自身
            all_neighbors = point_candidates[neighbor_indices]

            if np.all(x_center < all_neighbors[:, 0]):
                local_maxima.append(point_candidates[i])
        local_maxima = np.array(local_maxima)
    else:
        # throw a warning
        raise ValueError('No point in the region of interest')

    # determine the region of target part
    z = local_maxima[:, 2]
    z_min, z_max = np.min(z), np.max(z)
    z_interval = z_max - z_min
    # Select some points from each z region
    final_candidates = []
    num_candidates_in_each_level = [4,3,2,1]
    for z_level in range(1, len(num_candidates_in_each_level)+1): # 1, 2, 3, 4
        # Calculate the range for this level
        z_roi_min = z_min + z_interval * (z_level - 1) / 4
        z_roi_max = z_min + z_interval * z_level / 4
        this_level_candidates = local_maxima[(local_maxima[:, 2] >= z_roi_min) & (local_maxima[:, 2] < z_roi_max)]
        num_candidates_this_level = num_candidates_in_each_level[z_level-1]
        if len(this_level_candidates) > num_candidates_this_level:
            # if there are enough candidates in this level, randomly select N of them
            selected_candidates = this_level_candidates[np.random.choice(this_level_candidates.shape[0], num_candidates_in_each_level[z_level-1], replace=False)]
        else:
            # if there are not enough candidates in this level, select all of them
            selected_candidates = this_level_candidates
        
        if len(final_candidates) == 0:
            final_candidates = selected_candidates
        else:
            final_candidates = np.concatenate((final_candidates, selected_candidates), axis=0)

    # TODO: use dynamic model to determine the best grasp position
    # for now, just randomly sample one from the final candidates
    grasp_position = final_candidates[np.random.choice(len(final_candidates))]

    # find out the corresponding pc
    middle_line_xy = [0.0, 0.0]
    grasp_point_xy = grasp_position[0:2]
    # calculate the angle between the grasp position and the middle line
    z_angle = np.arctan2(grasp_point_xy[1] - middle_line_xy[1], grasp_point_xy[0] - middle_line_xy[0])

    plot_pc(pc_origin, grasp_position, z_angle, pc, local_maxima, final_candidates)
    
    # Calculate the transformation matrix
    rotation_matrix = np.array([[-np.sin(z_angle), 0.0, -np.cos(z_angle)],
                                [np.cos(z_angle), 0.0, -np.sin(z_angle)],
                                [0.0, -1.0, 0.0 ]])
    
    transform_matrix = np.identity(4)
    transform_matrix[:3, :3] = rotation_matrix
    transform_matrix[:3, 3] = grasp_position

    return transform_matrix



from competition_pipeline.utils import segment_cloth, filter_pc, plot_pc
import cv2
import numpy as np
import time
from sklearn.neighbors import NearestNeighbors

def sample_grasp(camera_pose_in_world,
                 camera_intrinsics,
                 image_rgb,
                 point_cloud):
    '''Give the observations required, return the grasp pose'''
    # Segment the cloth first
    mask = segment_cloth(image_rgb, camera_pose_in_world, camera_intrinsics)
    cv2.imwrite('../log/mask_comp.png', mask)
    print("Masking finished")

    # Get the point cloud
    pc = filter_pc(point_cloud, mask)
    
    z_offset = 0.3
    # sr_Y_min, sr_Y_max = self.Y_range
    # set region of interest (roi)
    pc = pc[pc[:, 0] < 0.2]
    x = pc[:, 0]
    y = pc[:, 1]
    z = pc[:, 2]
    sr_Y_min, sr_Y_max =  np.min(y), np.max(y)
    z_min, z_max = np.min(z), np.max(z)
    z_max = z_max - z_offset # manually set max height for grasping
    assert z_max > z_min, 'z_max should be larger than z_min'
    z_interval = z_max - z_min

    pool_part = np.array([1,2,3,4]) # 1 bottom part, 4 upper part
    # set random seed according to the current time
    np.random.seed(int(time.time()))
    target_part = np.random.choice(pool_part, 1, p=[0.4, 0.3, 0.2, 0.1])[0]
    # determine the region of target part
    z_roi_min = z_min + z_interval * (target_part - 1) / 4
    z_roi_max = z_min + z_interval * target_part / 4
    # print(f'z_roi: [{z_roi_min, z_roi_max}]')

    # sample a point with z in [z_roi_min, z_roi_max]
    # idx_arr = np.where((z >= z_roi_min) & (z <= z_roi_max))[0]

    # sample a point with z in [z_roi_min, z_roi_max] and y in [sr_Y_min, sr_Y_max]
    idx_arr = np.where((z >= z_roi_min) & (z <= z_roi_max) & (y >= sr_Y_min) & (y <= sr_Y_max))[0]
    if idx_arr is not None:

        point_candidates = pc[idx_arr]
        nbrs = NearestNeighbors(n_neighbors=4, algorithm='auto').fit(point_candidates)
        distances, indices = nbrs.kneighbors(point_candidates)

        local_maxima = []

        for i in range(point_candidates.shape[0]):
            x_center = point_candidates[i, 0]
            neighbor_indices = indices[i, 1:]  # 排除自身
            all_neighbors = point_candidates[neighbor_indices]

            if np.all(x_center > all_neighbors[:, 0]):
                local_maxima.append(point_candidates[i])

        local_maxima = np.array(local_maxima)

        if len(local_maxima) > 0:
            grasp_position = local_maxima[np.random.choice(local_maxima.shape[0])]
        else:
            # sample a point from local_maxima
            idx = np.random.choice(idx_arr)
            grasp_position = pc[idx]
    else:
        # throw a warning
        raise ValueError('No point in the region of interest')
    # find out the corresponding pc
    middle_line_xy = [0.0, 0.0]
    grasp_point_xy = grasp_position[0:2]
    # calculate the angle between the grasp position and the middle line
    z_angle = np.arctan2(grasp_point_xy[1] - middle_line_xy[1], grasp_point_xy[0] - middle_line_xy[0])

    plot_pc(pc, grasp_position, z_angle)

    pass
    
    # Calculate the grasp pose from grasp position and z_angle

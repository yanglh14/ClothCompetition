import os
from segment_anything import SamPredictor, sam_model_registry
import numpy as np
import cv2
import torch
import time

current_dir = os.path.dirname(__file__)
root_dir = os.path.dirname(current_dir)

image_size_width = 480
image_size_length = 960

def segment_cloth(image, camera_pose_in_world, camera_intrinsics, camera_resolution):
    # Crop the image first from the center
    image_col_start = int(camera_resolution[0] // 2 - image_size_width // 2)
    image_col_end = image_col_start + image_size_width
    image_row_start = int(camera_resolution[1] // 2 - image_size_length // 2)
    image_row_end = image_row_start + image_size_length

    image_cropped = image[image_row_start:image_row_end, image_col_start:image_col_end]

    path_to_checkpoint = os.path.join(root_dir, "ClothCompetition", "pth", "sam_vit_h_4b8939.pth")
    model_type = "vit_h"
    device = "cpu" # "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[model_type](checkpoint=path_to_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    predictor.set_image(image_cropped)
    # input_point = np.array([[345, 226]])
    # input_label = np.array([1])

    # Select two points in world coordinates
    # that are very likely to be on the cloth
    cloth_points = np.array([[0.0, 0.0, 0.8], [0.0, 0.0, 0.7]])

    # Calculate the corresponding pixel coordinates
    world_rotation_in_camera = camera_pose_in_world[:3, :3].T
    world_translation_in_camera = -world_rotation_in_camera.dot(camera_pose_in_world[:3, 3])
    image_points, _ = cv2.projectPoints(objectPoints=cloth_points, 
                             rvec=world_rotation_in_camera,
                             tvec=world_translation_in_camera, 
                             cameraMatrix=camera_intrinsics,
                             distCoeffs=None)
    
    # Convert the pixel coordinates to integers
    input_point = image_points.squeeze().astype(np.int32)
    # Offset the input point to the cropped image
    input_point = input_point - np.array([image_col_start, image_row_start])
    input_label = np.array([1, 1])
    mask, _, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=False,
    )
    # set the mask to 0 or 255
    mask = np.where(mask > 0.5, 255, 0).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    mask = mask[0]
    mask = cv2.erode(mask, kernel, iterations=5)

    # Draw the points on the image
    image_display = image_cropped.copy()
    for point in input_point:
        cv2.circle(image_display, tuple(point), 25, (0, 255, 0), -1)

    # Display the selected points
    cv2.namedWindow("selected_points", cv2.WINDOW_NORMAL)
    cv2.imshow("selected_points", image_display)
    cv2.waitKey()

    # Display the mask
    cv2.namedWindow("mask", cv2.WINDOW_NORMAL)
    cv2.imshow("mask", mask)
    cv2.waitKey()

    # Save image and mask
    current_dir = os.path.dirname(__file__)
    log_dir = os.path.join(current_dir, "log")
    current_time = int(time.time())
    cv2.imwrite(os.path.join(log_dir, "raw_image_{}.jpg".format(current_time)), image)
    cv2.imwrite(os.path.join(log_dir, "cropped_image_{}.jpg".format(current_time)), image_display)
    cv2.imwrite(os.path.join(log_dir, "mask_{}.jpg".format(current_time)), mask)  

    return mask

def get_pc_from_depth(depth_map, 
                      camera_intrinsics,
                      camera_pose_in_world,
                      mask,
                      ):
    # Get camera intrinsics elements
    fx = camera_intrinsics[0, 0]
    fy = camera_intrinsics[1, 1]
    cx = camera_intrinsics[0, 2]
    cy = camera_intrinsics[1, 2]

    # Generate the point cloud
    height, width = depth_map.shape

    # Create a meshgrid of pixel coordinates
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    Z = np.where(mask == 255, depth_map, 0)

    u, v, Z = u.flatten(), v.flatten(), Z.flatten()  # for depth_map, the value of the pixel is already in meters

    valid_indices = Z > 0
    u, v, Z = u[valid_indices], v[valid_indices], Z[valid_indices]

    # Compute the X, Y world coordinates
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy

    point_cloud = np.vstack((X, Y, Z)).transpose()
    points = transform_point_cloud(point_cloud, camera_pose_in_world)

    # voxelize the point cloud
    vox_pc = voxelize_pointcloud(points.astype(np.float32), voxel_size=0.0216)

    return vox_pc

def filter_pc(pc, mask, camera_resolution):
    # Crop the point clouds first from the center
    image_col_start = int(camera_resolution[0] // 2 - image_size_width // 2)
    image_col_end = image_col_start + image_size_width
    image_row_start = int(camera_resolution[1] // 2 - image_size_length // 2)
    image_row_end = image_row_start + image_size_length

    pc_cropped = pc[image_row_start:image_row_end, image_col_start:image_col_end]

    pc = pc_cropped[mask == 255]
    pc = voxelize_pointcloud(pc.astype(np.float32), voxel_size=0.0216)

    # Save point cloud to a file
    import pickle
    current_dir = os.path.dirname(__file__)
    log_dir = os.path.join(current_dir, "log")
    current_time = int(time.time())
    file_name = os.path.join(log_dir, "pc_{}".format(current_time))
    np.save(file_name, pc)

    return pc

def transform_point_cloud(point_cloud,
                          camera_pose_in_world):

    # Convert point cloud to homogeneous coordinates (add a row of 1's)
    ones = np.ones((point_cloud.shape[0], 1))
    points_homogeneous = np.hstack((point_cloud, ones))

    # Apply the transformation matrix to the point cloud
    point_cloud_transformed_homogeneous = points_homogeneous.dot(camera_pose_in_world)

    # Convert back from homogeneous coordinates by dropping the last column
    point_cloud_transformed = point_cloud_transformed_homogeneous[:, :3]

    return point_cloud_transformed

import pcl

def voxelize_pointcloud(pointcloud, voxel_size):
    cloud = pcl.PointCloud(pointcloud)
    sor = cloud.make_voxel_grid_filter()
    sor.set_leaf_size(voxel_size, voxel_size, voxel_size)
    pointcloud = sor.filter()
    pointcloud = np.asarray(pointcloud).astype(np.float32)
    return pointcloud

import matplotlib.pyplot as plt

def plot_pc(pc, grasp_position, z_angle):
    point_cloud = pc
    # First, convert your point cloud to a numpy array for easier manipulation
    point_cloud_np = np.array(point_cloud)

    # Split your NumPy array into positions (x, y, z) and colors (r, g, b)
    positions = point_cloud_np[:, :3]

    # Create a new matplotlib figure and axis.
    fig = plt.figure()
    # window size to be square
    fig.set_size_inches(10, 10)
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot using the x, y, and z coordinates and the color information
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c=[0, 0, 1], s=1)  # s is the size of the points

    # highlight the grasp position with idx
    ax.scatter(grasp_position[0], grasp_position[1], grasp_position[2], c=[1, 0, 0], s=3)
    # plot a arrow from the grasp point with angle
    ax.quiver(grasp_position[0], grasp_position[1], grasp_position[2], 0.1 * np.cos(z_angle), 0.1 * np.sin(z_angle), 0,
                color='red')

    ax.set_xlim3d(-0.5, 0.5)
    ax.set_ylim3d(-0.5, 0.5)
    ax.set_zlim3d(-0, 1.0)

    # Set labels for axes
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    # Show the plot
    plt.show()

import cv2
from competition_pipeline.sample_grasp import sample_grasp
import os
from cloth_tools.dataset.format import load_competition_observation

current_dir = os.path.dirname(__file__)

if __name__ == "__main__":
    # Read data from a specific folder
    data_folder = os.path.join(current_dir, "observation_start")

    observation = load_competition_observation(data_folder)

    # Read camera information
    camera_pose_in_world = observation.camera_pose_in_world
    camera_intrinsics = observation.camera_intrinsics
    camera_resolution = observation.camera_resolution

    # Read images from observations
    image_rgb = observation.image_left
    depth_map = observation.depth_map
    point_cloud = observation.point_cloud.points.reshape(camera_resolution[1], camera_resolution[0], 3)

    # Sample a grasp
    sample_grasp(camera_pose_in_world=camera_pose_in_world,
                 camera_intrinsics=camera_intrinsics,
                 image_rgb=image_rgb,
                 point_cloud=point_cloud)

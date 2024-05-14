import cv2
from competition_pipeline.sample_grasp import sample_grasp
import os
import matplotlib.pyplot as plt
from cloth_tools.dataset.download import download_latest_observation
from cloth_tools.dataset.format import load_competition_observation
from cloth_tools.visualization.opencv import draw_pose
from cloth_tools.dataset.upload import upload_grasp
# Save the grasp pose and upload it to the server
from airo_dataset_tools.data_parsers.pose import Pose
from airo_typing import HomogeneousMatrixType
from cloth_tools.dataset.bookkeeping import datetime_for_filename
from pathlib import Path
import json
import numpy as np
# from ClothCompetition.main_plan_e2e import Planner
from competition_pipeline.utils import init_segmentation
from ClothCompetition.main_plan_e2e import Planner
import time

current_dir = os.path.dirname(__file__)
current_dir = '/home/yang/dataset'
dataset_name = "cloth_competition_dataset_0000"
random_sample = True

if __name__ == "__main__":
    # Initialize predictor
    init_segmentation()
    planner = Planner()

    # planner = Planner()
    # dataset_dir = os.path.join(current_dir, dataset_name)
    dataset_dir = current_dir

    # Read all the directories in the dataset
    observation_dirs = [os.path.join(dataset_dir, d) for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    
    # Randomly shuffle the observation dirs if random_sample is True
    if random_sample:
        np.random.shuffle(observation_dirs)

    # Iterate through the observation directories
    for observation_dir in observation_dirs:
        time_start = time.time()
        observation_folder = os.path.join(observation_dir, "observation_start")
        print("data_dir: ", observation_dir)    
        observation = load_competition_observation(observation_folder)

        # Read camera information
        camera_pose_in_world = observation.camera_pose_in_world
        camera_intrinsics = observation.camera_intrinsics
        camera_resolution = observation.camera_resolution

        # Read images from observations
        image_rgb = observation.image_left
        depth_map = observation.depth_map
        point_cloud = observation.point_cloud.points.reshape(camera_resolution[1], camera_resolution[0], 3)

        # Sample a grasp
        grasp_pose = sample_grasp(
            camera_pose_in_world=camera_pose_in_world,
            camera_intrinsics=camera_intrinsics,
            camera_resolution=camera_resolution,
            image_rgb=image_rgb,
            point_cloud=point_cloud,
            planner = planner,
        )
        grasp_pose = grasp_pose[0]
        # Give the pose an offset along the TCP frame z axis
        offset = np.array([0.0, 0.0, 0.05, 1.0])
        new_position = np.matmul(grasp_pose, offset)
        grasp_pose[:3, 3] = new_position[:3]

        # Visualize the grasp
        image_bgr = cv2.cvtColor(observation.image_left, cv2.COLOR_RGB2BGR)
        draw_pose(image_bgr, grasp_pose, camera_intrinsics, camera_pose_in_world, 0.1)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        print("Time cost for sampling grasp: ", time.time()-time_start)
        plt.figure(figsize=(10, 5))
        plt.imshow(image_rgb)
        plt.title("Example grasp pose")
        plt.savefig(os.path.join(current_dir, "grasp_pose.png"))
        plt.show()

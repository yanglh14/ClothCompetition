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
import time
from ClothCompetition.main_plan_e2e import Planner

current_dir = os.path.dirname(__file__)

in_competition = False

if __name__ == "__main__":
    planner = Planner()
    if in_competition:
        # TODO: ask for IP address
        # Read data from the competition server
        server_url = "http://10.42.0.1:5000"
        data_dir = Path("data")
        dataset_dir = data_dir / "downloaded_dataset_0000"
        observation_dir, sample_id = download_latest_observation(dataset_dir, server_url)
    else:
        # Read data from a specific folder
        observation_dir = os.path.join(current_dir, "sample_000000", "observation_start")
    time_start= time.time()
    observation = load_competition_observation(observation_dir)
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

    # Give the pose an offset along the TCP frame z axis
    offset = np.array([0.0, 0.0, 0.05, 1.0])
    new_position = np.matmul(grasp_pose, offset)
    grasp_pose[:3, 3] = new_position[:3]

    # Visualize the grasp
    image_bgr = cv2.cvtColor(observation.image_left, cv2.COLOR_RGB2BGR)
    draw_pose(image_bgr, grasp_pose, camera_intrinsics, camera_pose_in_world, 0.1)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(10, 5))
    plt.imshow(image_rgb)
    plt.title("Example grasp pose")
    plt.savefig(os.path.join(current_dir, "grasp_pose.png"))
    plt.show()

    def save_grasp_pose(grasps_dir: str, grasp_pose_fixed: HomogeneousMatrixType) -> str:
        os.makedirs(grasps_dir, exist_ok=True)

        grasp_pose_name = f"grasp_pose_{datetime_for_filename()}.json"
        grasp_pose_file = os.path.join(grasps_dir, grasp_pose_name)

        with open(grasp_pose_file, "w") as f:
            grasp_pose_model = Pose.from_homogeneous_matrix(grasp_pose_fixed)
            json.dump(grasp_pose_model.model_dump(exclude_none=False), f, indent=4)

        return grasp_pose_file
    
    if in_competition:
        grasps_dir = f"data/grasps_{sample_id}"
    else:
        grasps_dir = os.path.join(current_dir, "grasps_test")

    grasp_pose_file = save_grasp_pose(grasps_dir, grasp_pose)
    
    if in_competition:
        team_name = "test_greater_bay"
        upload_grasp(grasp_pose_file, team_name, sample_id, server_url) 
    print('time cost:', time.time()-time_start)

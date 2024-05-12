from ClothCompetition.real_robot.utils.rs_utils import segment_cloth
from pathlib import Path
from cloth_tools.dataset.download import download_latest_observation
from cloth_tools.dataset.format import load_competition_observation
import cv2

server_url = "https://robotlab.ugent.be"
data_dir = Path("data")
dataset_dir = data_dir / "downloaded_dataset_0000"

if __name__ == "__main__":
    observation_dir, sample_id = download_latest_observation(dataset_dir, server_url)
    observation = load_competition_observation(observation_dir)

    # Get camera infos
    camera_pose_in_world = observation.camera_pose_in_world
    camera_intrinsics = observation.camera_intrinsics

    # Get RGB and Depth image from the server
    image_bgr = cv2.cvtColor(observation.image_left, cv2.COLOR_RGB2BGR)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_depth = observation.depth_image

    # download RGB image and Depth image from the server
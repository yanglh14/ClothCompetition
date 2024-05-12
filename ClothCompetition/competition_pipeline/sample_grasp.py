from ClothCompetition.real_robot.utils.rs_utils import segment_cloth

def get_images_from_server():
    # TODO: use the code provided by the competition
    # For now just read a sample image
    pass

def get_camera_intrinsics():
    pass

if __name__ == "__main__":
    # download RGB image and Depth image from the server
    rgb, depth = get_images_from_server()

    # Segment the cloth and get the mask
    # mask = segment_cloth(image)
import numpy as np
import cv2
from ClothCompetition.real_robot.utils.euler import euler2mat

import yaml
import os

# get file path
path = os.path.dirname(os.path.abspath(__file__))

with open(path + '/../cfg/robots.yaml', 'r') as file:
    config = yaml.safe_load(file)

camera_to_robot_left_trans = config['camera_to_robot_left']['translation']
camera_to_robot_left_rpy = config['camera_to_robot_left']['rotation']
robot_left_to_origin_trans = config['robot_left']['robot_to_origin']['translation']
robot_left_to_origin_rpy = config['robot_left']['robot_to_origin']['rotation']


# Define the range of yellow color in HSV
yellow_lower = np.array([20, 100, 100], dtype="uint8")
yellow_upper = np.array([30, 255, 255], dtype="uint8")

cloth_lower = np.array([80, 80, 40], dtype="uint8")
cloth_upper = np.array([140, 140, 160], dtype="uint8")

orange_lower = np.array([0, 100, 100], dtype="uint8")
orange_upper = np.array([40, 255, 255], dtype="uint8")

clothes_color_range = {
    'green_leaf': np.array([0, 100, 100, 30, 255, 255], dtype="uint8"),
}

def compute_transformation_matrix(translation, rpy, radians=True):
    # Convert RPY angles from degrees to radians
    if not radians:
        rpy = np.radians(rpy)

    rotation_matrix = euler2mat(*rpy)
    # Create the transformation matrix
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = rotation_matrix
    transform_matrix[:3, 3] = translation

    return transform_matrix

camera_to_robot_left = compute_transformation_matrix(camera_to_robot_left_trans, camera_to_robot_left_rpy)
robot_left_to_origin = compute_transformation_matrix(robot_left_to_origin_trans, robot_left_to_origin_rpy)
matrix_camera_to_world = robot_left_to_origin.dot(camera_to_robot_left)

def object_detection(image):

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # save h,s,v three files
    # cv2.imwrite('../log/h.png', hsv[:, :, 0])
    # cv2.imwrite('../log/s.png', hsv[:, :, 1])
    # cv2.imwrite('../log/v.png', hsv[:, :, 2])

    # Create a mask for the colth
    mask = cv2.inRange(hsv, cloth_lower, cloth_upper)
    # cv2.imwrite('../log/mask.png', mask)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        c = max(contours, key=cv2.contourArea)

        # Determine the center of the contour
        M = cv2.moments(c)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0

        # Draw the contour and centroid on the image
        # cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
        # cv2.circle(image, (cX, cY), 5, (255, 255, 255), -1)
        # cv2.imwrite('../log/contour.png', image)

    else:
        raise Exception("No cloth found in the image.")

    mask = np.zeros(image.shape[:2], dtype="uint8")
    cv2.drawContours(mask, [c], -1, color=255, thickness=-1)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=3)

    return mask

def get_cloth_range(cloth_name):
    if cloth_name == 'green_leaf':
        range_temp = clothes_color_range[cloth_name]
        return range_temp[0:3], range_temp[3:6]
    else:
        raise Exception("Invalid cloth name.")

# for cloth competition's purpose
def clothes_detection(image,cloth_name):

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # save h,s,v three files
    # cv2.imwrite('../log/h.png', hsv[:, :, 0])
    # cv2.imwrite('../log/s.png', hsv[:, :, 1])
    # cv2.imwrite('../log/v.png', hsv[:, :, 2])

    cur_cloth_lower, cur_cloth_upper = get_cloth_range(cloth_name)
    # Create a mask for the colth
    mask = cv2.inRange(hsv, cur_cloth_lower, cur_cloth_upper)
    # cv2.imwrite('../log/mask.png', mask)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        c = max(contours, key=cv2.contourArea)

        # Determine the center of the contour
        M = cv2.moments(c)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0

        # Draw the contour and centroid on the image
        cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
        cv2.circle(image, (cX, cY), 5, (255, 255, 255), -1)
        cv2.imwrite('../log/contour_comp.png', image)

    else:
        raise Exception("No cloth found in the image.")

    mask = np.zeros(image.shape[:2], dtype="uint8")
    cv2.drawContours(mask, [c], -1, color=255, thickness=-1)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=3)

    return mask

def transform_point_cloud(point_cloud):

    # Convert point cloud to homogeneous coordinates (add a row of 1's)
    ones = np.ones((point_cloud.shape[0], 1))
    points_homogeneous = np.hstack((point_cloud, ones))

    # Apply the transformation matrix to the point cloud
    point_cloud_transformed_homogeneous = points_homogeneous.dot(matrix_camera_to_world.T)

    # Convert back from homogeneous coordinates by dropping the last column
    point_cloud_transformed = point_cloud_transformed_homogeneous[:, :3]

    return point_cloud_transformed

import pcl


# def get_partial_particle(full_particle, observable_idx):
#     return np.array(full_particle[observable_idx], dtype=np.float32)


def voxelize_pointcloud(pointcloud, voxel_size):
    cloud = pcl.PointCloud(pointcloud)
    sor = cloud.make_voxel_grid_filter()
    sor.set_leaf_size(voxel_size, voxel_size, voxel_size)
    pointcloud = sor.filter()
    pointcloud = np.asarray(pointcloud).astype(np.float32)
    return pointcloud


if __name__ == '__main__':
    # # Load the image
    # image = cv2.imread('../../log/rgb.png')
    #
    # mask = object_detection(image)
    #
    # # apply the mask to the image to see the result
    # # This will leave only the object, making all other pixels black
    # result = cv2.bitwise_and(image, image, mask=mask)
    #
    # # Show the mask and the result
    # cv2.imshow('Mask', mask)
    # cv2.imshow('Result', result)
    #
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    point_cloud = np.load('../../log/rs_data.npy')
    point_cloud = transform_point_cloud(point_cloud)
    np.save('../../real_exp/log/transformed_pc.npy', point_cloud)
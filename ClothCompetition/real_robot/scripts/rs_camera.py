import rospy
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import message_filters

from ClothCompetition.real_robot.utils.rs_utils import object_detection, transform_point_cloud, voxelize_pointcloud
class RSListener:
    def __init__(self):

        # rospy.init_node('rs_listener', anonymous=True)
        self.bridge = CvBridge()

        self.mask = None
        self.points = None
        self.vox_pc = None
        self.image = None

        self.image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self._image_callback)

        # Use message_filters to subscribe to the image and camera info topics
        depth_image_sub = message_filters.Subscriber('/camera/aligned_depth_to_color/image_raw', Image)
        camera_info_sub = message_filters.Subscriber('/camera/aligned_depth_to_color/camera_info', CameraInfo)

        ts = message_filters.TimeSynchronizer([depth_image_sub, camera_info_sub], 10)
        ts.registerCallback(self._depth_callback)

        # try:
        #     rospy.spin()
        # except KeyboardInterrupt or rospy.is_shutdown():
        #     print("Shutting down depth image reader node.")

    def _image_callback(self, data):
        try:
            # Convert the image to OpenCV format
            self.image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
        # cv2.imwrite('../log/rgb.png', image)

        self.mask = object_detection(self.image)

    def _depth_callback(self, depth_image_msg, camera_info_msg):

        try:
            if self.mask is None:
                return
            # Convert the ROS image to OpenCV format using a cv_bridge helper function
            depth_image = self.bridge.imgmsg_to_cv2(depth_image_msg, depth_image_msg.encoding)
            self.depth_image = depth_image.copy()

            # Get the camera intrinsic parameters
            self.fx = camera_info_msg.K[0]
            self.fy = camera_info_msg.K[4]
            self.cx = camera_info_msg.K[2]
            self.cy = camera_info_msg.K[5]

            # Generate the point cloud
            height, width = depth_image.shape

            # Create a meshgrid of pixel coordinates
            u, v = np.meshgrid(np.arange(width), np.arange(height))

            # Apply the mask to the depth image
            # Only consider pixels where the mask is 255
            Z = np.where(self.mask == 255, depth_image, 0)
            # import cv2
            # np.save('../log/masked_depth.npy', Z)
            # np.save('../log/depth.npy', depth_image)
            # cv2.imwrite('../log/mask.png', self.mask)

            # Flatten the arrays for vectorized computation
            u, v, Z = u.flatten(), v.flatten(), Z.flatten() * 0.001  # Depth scale (mm to meters)

            # Filter out the points with zero depth after masking
            valid_indices = Z > 0
            u, v, Z = u[valid_indices], v[valid_indices], Z[valid_indices]

            # Compute the X, Y world coordinates
            X = (u - self.cx) * Z / self.fx
            Y = (v - self.cy) * Z / self.fy

            # Stack the coordinates into a point cloud
            point_cloud = np.vstack((X, Y, Z)).transpose()
            # convert camera coordinate to robot coordinate
            points = transform_point_cloud(point_cloud)
            # voxelize the point cloud
            self.vox_pc = voxelize_pointcloud(points.astype(np.float32), voxel_size=0.0216)
            # np.save('../log/transformed_pc.npy', points)
            # np.save('../log/vox_pc.npy', self.vox_pc)
            # np.save('../log/masked_pc.npy', point_cloud)

        except CvBridgeError as e:
            print(e)

    def get_pc_given_mask(self, mask):
        if self.depth_image is None:
            return None
        depth_image = self.depth_image.copy()
        # Generate the point cloud
        height, width = depth_image.shape

        # Create a meshgrid of pixel coordinates
        u, v = np.meshgrid(np.arange(width), np.arange(height))

        Z = np.where(mask == 255, depth_image, 0)

        u, v, Z = u.flatten(), v.flatten(), Z.flatten() * 0.001  # Depth scale (mm to meters)

        # Filter out the points with zero depth after masking
        valid_indices = Z > 0
        u, v, Z = u[valid_indices], v[valid_indices], Z[valid_indices]

        # Compute the X, Y world coordinates
        X = (u - self.cx) * Z / self.fx
        Y = (v - self.cy) * Z / self.fy

        # Stack the coordinates into a point cloud
        point_cloud = np.vstack((X, Y, Z)).transpose()
        # convert camera coordinate to robot coordinate
        points = transform_point_cloud(point_cloud)
        # voxelize the point cloud
        vox_pc = voxelize_pointcloud(points.astype(np.float32), voxel_size=0.0216)

        return vox_pc

    def get_pc_given_pixel(self, pixels):
        depth_image = self.depth_image.copy()
        point_cloud = []
        signal = [[-1, +1], [+1, +1], [+1, -1], [-1, -1]]
        for i, pixel in enumerate(pixels):
            u, v = pixel
            Z = depth_image[v, u] * 0.001
            while Z ==0:
                u = u+signal[i][0]
                v = v+signal[i][1]
                Z = depth_image[u, v] * 0.001
            X = (u - self.cx) * Z / self.fx
            Y = (v - self.cy) * Z / self.fy

            point_cloud.append([X, Y, Z])
        points = transform_point_cloud(np.array(point_cloud))

        return points

if __name__ == '__main__':
    rospy.init_node('rs_camera')
    rs_listener = RSListener()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down depth image reader node.")
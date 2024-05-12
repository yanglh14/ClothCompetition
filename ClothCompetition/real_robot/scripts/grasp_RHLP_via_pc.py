# script to segment the cloth and grasp it
# RHLP: right robot to hold the cloth, left robot to pick the cloth
# via point cloud
#
# Zhang Zeqing
# 2025/05/08

import cv2
import time
import numpy as np
from random import sample
import rospy
import os
import yaml
import threading
from ClothCompetition.real_robot.scripts.robot import Robot
from ClothCompetition.real_robot.scripts.env import EnvReal
from ClothCompetition.real_robot.scripts.rs_camera import RSListener
from ClothCompetition.real_robot.utils.rs_utils import clothes_detection,mask_no_rgb,segment_cloth
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

class Grasp:
    def __init__(self):
        self.env = EnvReal()
        self.rs_listener = RSListener()

        # def safe region (y-z plane) for legt Robot to grasp
        offset_piker = self.env.robot_right.picker_to_ee_trans[2]
        sr_mid_y = self.env.robot_right.init_pose[0]+offset_piker
        sr_mid_Y = self.env.robot_right.robot_to_origin_trans[1]-sr_mid_y
        sr_Y_max = sr_mid_Y + 0.08
        sr_Y_min = sr_mid_Y - 0.08
        self.Y_range = [sr_Y_min, sr_Y_max]

    def get_image(self):
        if self.rs_listener.image is None:
            time.sleep(0.1)
        return self.rs_listener.image.copy()

    def sample_grasp_posi(self, pc):
        z_offset = 0.3
        # sr_Y_min, sr_Y_max = self.Y_range
        # set region of interest (roi)
        pc = pc[pc[:, 0] > 0]
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
        # # find out the corresponding pc
        # grasp_position = pc[idx]
        # middle_line_xy = self.env.robot_right.get_picker_pose_in_origin()[0:2]
        middle_line_xy = self.env.robot_right.get_ee_pose_in_origin()[0][0:2] + np.array([0, -0.165])
        grasp_point_xy = grasp_position[0:2]
        # calculate the angle between the grasp position and the middle line
        z_angle = np.arctan2(grasp_point_xy[1] - middle_line_xy[1], grasp_point_xy[0] - middle_line_xy[0])

        self.plot_pc(pc, grasp_position, z_angle)
        return grasp_position, z_angle

    def plot_pc(self, pc, grasp_position, z_angle):
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

        ax.set_xlim3d(-0, 1.0)
        ax.set_ylim3d(-0.5, 0.5)
        ax.set_zlim3d(-0, 1.0)

        # Set labels for axes
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        # Show the plot
        plt.show()

    def generate_stretch_traj4TCP(self, POSI_start_L, POSI_start_R, L, N, dur):
        # generate stretch trajectories for TCP of two arms
        X0_L, Y0_L, Z0_L = POSI_start_L
        X0_R, Y0_R, Z0_R = POSI_start_R
        X_R_ls, Y_R_ls, Z_R_ls, X_L_ls, Y_L_ls, Z_L_ls = [], [], [], [], [], []
        # time_from_start_ls = []
        # time_from_start = 0.0
        dt = dur / N

        for i in range(N + 1):
            # time_from_start += dur / N
            # time_from_start_ls.append(time_from_start)

            X_L = X0_L
            Y_L = Y0_L - i * L / (2 * N)
            Z_L = Z0_L
            theta = i * np.pi / (2 * N)
            X_R = X0_R
            Y_R = Y_L + L * np.sin(theta)
            Z_R = Z_L - L * np.cos(theta)

            X_L_ls.append(X_L)
            Y_L_ls.append(Y_L)
            Z_L_ls.append(Z_L)
            X_R_ls.append(X_R)
            Y_R_ls.append(Y_R)
            Z_R_ls.append(Z_R)
        # np.array of [X_L_ls, Y_L_ls, Z_L_ls] with 3 columns
        action_L = np.array([X_L_ls, Y_L_ls, Z_L_ls]).T
        action_R = np.array([X_R_ls, Y_R_ls, Z_R_ls]).T
        return action_L, action_R, dt
    def get_edge(self):
        image = self.get_image()
        # Convert to graycsale
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Blur the image for better edge detection
        img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)

        # Sobel Edge Detection
        sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5)  # Sobel Edge Detection on the X axis
        sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5)  # Sobel Edge Detection on the Y axis
        sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1,
                            ksize=5)  # Combined X and Y Sobel Edge Detection
        # # Display Sobel Edge Detection Images
        # cv2.imshow('Sobel X', sobelx)
        # cv2.waitKey(0)
        # cv2.imshow('Sobel Y', sobely)
        # cv2.waitKey(0)
        # cv2.imshow('Sobel X Y using Sobel() function', sobelxy)
        # cv2.waitKey(0)

        # Canny Edge Detection
        edges = cv2.Canny(image=img_blur, threshold1=1, threshold2=200)  # Canny Edge Detection
        # save Canny Edge Detection Image
        cv2.imwrite('../log/edges.png', edges)
        # cv2.waitKey(0)

        cv2.destroyAllWindows()
        return edges

if __name__ == '__main__':
    gp = Grasp()
    # gp.env.gripper_open()
    gp.env.robot_left.set_gripper_open(True)
    gp.env.reset()
    gp.env.gripper_close()
    gp.env.robot_left.set_gripper_open(True)
    print("Initialization finished")

    isTest = False
    if isTest == False:
        ## get the mask of the cloth
        image = gp.get_image()

        ## method1: using RGB for data collection
        # mask = clothes_detection(image,'green_leaf')
        ## method2: mask the given area
        # mask = mask_no_rgb(image)
        ## method3: using "segment_anything"
        mask = segment_cloth(image)
        cv2.imwrite('../log/mask_comp.png', mask)
        print("Masking finished")

        ## get the vox_pc of the mask
        cloth_pc = gp.rs_listener.get_pc_given_mask(mask)
        # # print('cloth_pc:', cloth_pc.shape)
        # np.save('../log/cloth_pc.npy', cloth_pc)
        # # save mask
        # cv2.imwrite('../log/mask.png', mask)
        # # filter the image with mask
        # image_filtered = cv2.bitwise_and(image, image, mask=mask)
        # cv2.imwrite('../log/image_filtered.png', image_filtered)

        ## sample a grasp POSItion from the point cloud (world frame)
        grasp_POSI, z_angle = gp.sample_grasp_posi(cloth_pc) # rad
        print("Grasp pose sampled")
    else:
        ## given a goal position (testing purpose)
        right_piker_init_POSI = gp.env.robot_right.get_ee_pose_in_origin()[0] + np.array([0.0, -0.165, 0.0])
        grasp_POSI = right_piker_init_POSI + np.array([0.0, -0.1, -0.3])
        z_angle = 0
    left_piker_cur_POSI = np.round(gp.env.robot_left.get_ee_pose_in_origin()[0] + np.array([-0.165, 0.0, 0.0]), 8)
    print('cur_piker_POSI:', left_piker_cur_POSI)
    print('grasp_POSI:', grasp_POSI, 'z_angle:', z_angle)

    ## move L arm to the grasp pose (with initial orientation)(world frame)
    gp.env.move_L_arm_steps(grasp_POSI, z_angle)

    ## close the gripper
    gp.env.robot_left.set_gripper_open(False)

    # distance in z-axis of two grippers
    z_dist = np.abs(gp.env.robot_left.get_ee_pose_in_origin()[0][2] - gp.env.robot_right.get_ee_pose_in_origin()[0][2])
    print('z_dist:', z_dist)
    # z_dist=0.4 #<<<< testing purpose

    ## Mehotd 1: using 'step' func.
    # POSI_start_L = gp.env.robot_left.get_ee_pose_in_origin()[0] + np.array([0, 0.165, 0]) # TCP
    # POSI_start_R = gp.env.robot_right.get_ee_pose_in_origin()[0] - np.array([0.165, 0, 0]) # TCP
    # actionL, actionR, dt = gp.generate_stretch_traj4TCP(POSI_start_L, POSI_start_R, z_dist, 50, 5)
    # actions = np.concatenate((actionR, actionL), axis=1) # 6 columns
    # gp.env.step_stretch(actions, dt)

    ## Method2: using 'prepare_move' func.
    POSI_end_L = gp.env.robot_left.get_ee_pose_in_origin()[0] + np.array([0, 0.165, 0]) + np.array([0.4,-z_dist*0.8,0]) # TCP
    # POSI_end_R = gp.env.robot_right.get_ee_pose_in_origin()[0] - np.array([0.165, 0, 0]) + np.array([0,0.0,0.7]) # TCP
    # gp.env.move_arm(POSI_end_L,None,10)

    stretch_dist_per_arm = z_dist/2  # min: 0.15, max: 0.4
    if stretch_dist_per_arm < 0.15:
        stretch_dist_per_arm = 0.15
    elif stretch_dist_per_arm > 0.4:
        stretch_dist_per_arm = 0.4

    stretch_dist4LR = stretch_dist_per_arm  # min: 0.15, max: 0.4
    stretch_dist4RR = stretch_dist_per_arm  # min: 0.15, max: 0.4
    gp.env.move_L_arm_stretch(stretch_dist4LR, 10)
    gp.env.move_R_arm_stretch(stretch_dist4RR, 10)



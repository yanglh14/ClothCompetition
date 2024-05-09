# script to segment the cloth and grasp it
# RHLP: right robot to hold the cloth, left robot to pick the cloth
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
from ClothCompetition.real_robot.utils.rs_utils import clothes_detection

class Grasp:
    def __init__(self):
        self.env = EnvReal()
        self.rs_listener = RSListener()

        # def safe region (y-z plane) for Right Robot to grasp
        offset_piker = self.env.robot_left.picker_to_ee_trans[2]
        sr_mid_y = self.env.robot_left.init_pose[0]+offset_piker
        sr_mid_Y = self.env.robot_left.robot_to_origin_trans[1]+sr_mid_y
        sr_Y_max = sr_mid_Y + 0.08
        sr_Y_min = sr_mid_Y - 0.08
        self.Y_range = [sr_Y_min, sr_Y_max]

    def get_image(self):
        if self.rs_listener.image is None:
            time.sleep(0.1)
        return self.rs_listener.image.copy()

    def sample_grasp_posi(self, pc):
        z_offset = 0.3
        sr_Y_min, sr_Y_max = self.Y_range
        # set region of interest (roi)
        x = pc[:, 0]
        y = pc[:, 1]
        z = pc[:, 2]
        z_min, z_max = np.min(z), np.max(z)
        z_max = z_max - z_offset # manually set max height for grasping
        z_interval = z_max - z_min

        pool_part = np.array([1,2,3,4])
        target_part = np.random.choice(pool_part, 1, p=[0.05, 0.1, 0.2, 0.65])[0]
        # determine the region of target part
        z_roi_min = z_min + z_interval * (target_part - 1) / 4
        z_roi_max = z_min + z_interval * target_part / 4
        # print(f'z_roi: [{z_roi_min, z_roi_max}]')

        # sample a point with z in [z_roi_min, z_roi_max]
        # idx_arr = np.where((z >= z_roi_min) & (z <= z_roi_max))[0]

        # sample a point with z in [z_roi_min, z_roi_max] and y in [sr_Y_min, sr_Y_max]
        idx_arr = np.where((z >= z_roi_min) & (z <= z_roi_max) & (y >= sr_Y_min) & (y <= sr_Y_max))[0]
        if idx_arr is not None:
            # sample a point from idx_ls
            idx = np.random.choice(idx_arr)
        else:
            # throw a warning
            raise ValueError('No point in the region of interest')
        # find out the corresponding pc
        grasp_position = pc[idx]
        return grasp_position

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

if __name__ == '__main__':
    gp = Grasp()
    gp.env.gripper_open()
    gp.env.reset()
    gp.env.gripper_close()
    gp.env.robot_right.set_gripper_open(True)
    print("Initialization finished")
    print("init posi {rr}:", gp.env.robot_right.init_pose[0:3])

    ## get the mask of the cloth
    # image = gp.get_image()
    # mask = clothes_detection(image,'green_leaf')
    # # cv2.imwrite('../log/mask_comp.png', mask)

    ## get the vox_pc of the mask
    # cloth_pc = gp.rs_listener.get_pc_given_mask(mask)
    # # print('cloth_pc:', cloth_pc.shape)

    # sample a grasp POSItion from the point cloud (world frame)
    # grasp_POSI = gp.sample_grasp_posi(cloth_pc)

    # given a goal position (testing purpose)
    right_piker_init_POSI = gp.env.robot_right.get_ee_pose_in_origin()[0] + np.array([0.0, -0.165, 0.0])
    grasp_POSI = right_piker_init_POSI + np.array([0.0, -0.1, -0.3])

    left_piker_cur_POSI = np.round(gp.env.robot_left.get_ee_pose_in_origin()[0] + np.array([-0.165, 0.0, 0.0]), 8)
    print('cur_piker_POSI:', left_piker_cur_POSI)
    print('grasp_POSI:', grasp_POSI)

    ## move R arm to the grasp pose (with initial orientation)(world frame)
    # # gp.env.move(goal_POSI, arm='right')
    # # gp.env.move_R_arm(goal_POSI)
    # gp.env.move_R_arm_steps(grasp_POSI)

    ## move L arm to the grasp pose (with initial orientation)(world frame)
    # gp.env.move_L_arm_steps(grasp_POSI)

    ## close the gripper
    # gp.env.robot_right.set_gripper_open(False)

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

    stretch_dist4LR = 0.2  # min: 0.15, max: 0.4
    stretch_dist4RR = 0.2  # min: 0.15, max: 0.4
    gp.env.move_L_arm_stretch(stretch_dist4LR,10)
    gp.env.move_R_arm_stretch(stretch_dist4RR, 10)





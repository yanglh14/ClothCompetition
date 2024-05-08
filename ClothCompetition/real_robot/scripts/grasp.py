# script to segment the cloth and grasp it
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

    def get_image(self):
        if self.rs_listener.image is None:
            time.sleep(0.1)
        return self.rs_listener.image.copy()

    def sample_grasp_posi(self, pc):
        z_offset = 0.1
        # set region of interest (roi)
        x = pc[:, 0]
        y = pc[:, 1]
        z = pc[:, 2]
        z_min, z_max = np.min(z), np.max(z)
        z_max = z_max - z_offset # manually set max height for grasping
        z_interval = z_max - z_min
        pool_part = [1,2,2,3,3,3,4,4,4,4]
        # sample a point from pool_part
        target_part = sample(pool_part, 1)[0]
        # determine the region of target part
        z_roi_min = z_min + z_interval * (target_part - 1) / 4
        z_roi_max = z_min + z_interval * target_part / 4
        print(f'z_roi: [{z_roi_min, z_roi_max}]')
        # sample a point with z in [z_roi_min, z_roi_max]
        idx_arr = np.where((z >= z_roi_min) & (z <= z_roi_max))[0]
        if idx_arr is not None:
            # sample a point from idx_ls
            idx = np.random.choice(idx_arr)
        else:
            # throw a warning
            raise ValueError('No point in the region of interest')
        # find out the corresponding pc
        grasp_position = pc[idx]
        return grasp_position



if __name__ == '__main__':
    gp = Grasp()
    gp.env.reset()
    gp.env.gripper_close()
    gp.env.robot_right.set_gripper_open(True)
    image = gp.get_image()
    mask = clothes_detection(image,'green_leaf')
    # cv2.imwrite('../log/mask_comp.png', mask)
    ## get the vox_pc of the mask
    cloth_pc = gp.rs_listener.get_pc_given_mask(mask)
    print('cloth_pc:', cloth_pc.shape)

    # sample a grasp position from the point cloud
    grasp_posi = gp.sample_grasp_posi(cloth_pc)

    # move R arm to the grasp pose (with initial orientation)
    gp.env.move_R_arm(grasp_posi)




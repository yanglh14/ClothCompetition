# script to segment the cloth and grasp it
#
# Zhang Zeqing
# 2025/05/08

import cv2
import time
import numpy as np
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


if __name__ == '__main__':
    gp = Grasp()
    gp.env.reset()
    gp.env.gripper_close()
    image = gp.get_image()
    mask = clothes_detection(image,'green_leaf')
    # cv2.imwrite('../log/mask_comp.png', mask)
    ## get the vox_pc of the mask
    cloth_pc = gp.rs_listener.get_pc_given_mask(mask)




# script to segment the cloth and grasp it
#
# Zhang Zeqing
# 2025/05/08

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
        return self.rs_listener.image

    def get_mask(self):
        self.mask = clothes_detection(self.image)


if __name__ == '__main__':
    grasp = Grasp()
    image = grasp.get_image()
    mask = clothes_detection(image,'green_leaf')




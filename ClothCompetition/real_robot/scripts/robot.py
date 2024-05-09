import sys
import threading

import rospy
import actionlib
from controller_manager_msgs.srv import SwitchControllerRequest, SwitchController
from controller_manager_msgs.srv import LoadControllerRequest, LoadController
from controller_manager_msgs.srv import ListControllers, ListControllersRequest
import geometry_msgs.msg as geometry_msgs
from cartesian_control_msgs.msg import (
    FollowCartesianTrajectoryAction,
    FollowCartesianTrajectoryGoal,
    CartesianTrajectoryPoint,
)
from ClothCompetition.real_robot.utils.euler import quat2euler, euler2quat
from ClothCompetition.real_robot.utils.rs_utils import compute_transformation_matrix
import tf
import numpy as np
import os
import yaml
from std_msgs.msg import Bool
import time

# All of those controllers can be used to execute joint-based trajectories.
# The scaled versions should be preferred over the non-scaled versions.
JOINT_TRAJECTORY_CONTROLLERS = [
    "scaled_pos_joint_traj_controller",
    "scaled_vel_joint_traj_controller",
    "pos_joint_traj_controller",
    "vel_joint_traj_controller",
    "forward_joint_traj_controller",
]

# All of those controllers can be used to execute Cartesian trajectories.
# The scaled versions should be preferred over the non-scaled versions.
CARTESIAN_TRAJECTORY_CONTROLLERS = [
    "pose_based_cartesian_traj_controller",
    "joint_based_cartesian_traj_controller",
    "forward_cartesian_traj_controller",
]

# We'll have to make sure that none of these controllers are running, as they will
# be conflicting with the joint trajectory controllers
CONFLICTING_CONTROLLERS = ["joint_group_vel_controller", "twist_controller"]
class Robot:

    def __init__(self, config, **kwargs):
        self.robot_name = config['robot_name']

        timeout = rospy.Duration(5)
        self.switch_srv = rospy.ServiceProxy(
            "{}/controller_manager/switch_controller".format(self.robot_name), SwitchController
        )
        self.load_srv = rospy.ServiceProxy("{}/controller_manager/load_controller".format(self.robot_name), LoadController)
        self.list_srv = rospy.ServiceProxy("{}/controller_manager/list_controllers".format(self.robot_name), ListControllers)
        try:
            self.switch_srv.wait_for_service(timeout.to_sec())
        except rospy.exceptions.ROSException as err:
            rospy.logerr("Could not reach controller switch service. Msg: {}".format(err))
            sys.exit(-1)

        self.joint_trajectory_controller = JOINT_TRAJECTORY_CONTROLLERS[0]
        self.cartesian_trajectory_controller = CARTESIAN_TRAJECTORY_CONTROLLERS[1]

        self.trajectory_client = actionlib.SimpleActionClient(
            "{}/{}/follow_cartesian_trajectory".format(self.robot_name, self.cartesian_trajectory_controller),
            FollowCartesianTrajectoryAction,
        )

        # if need to switch controller
        self.switch_controller(self.cartesian_trajectory_controller)

        # Wait for action server to be ready
        timeout = rospy.Duration(5)
        if not self.trajectory_client.wait_for_server(timeout):
            rospy.logerr("Could not reach controller action server.")
            sys.exit(-1)

        self.tf_listener = tf.TransformListener()
        self.pub = rospy.Publisher('/{}_gripper/set_gripper_open'.format(self.robot_name), Bool, queue_size=1)

        self.moving = False

        self.trajectory_log = []
        self.init_pose = config['init_pose']

        self.dt = 0.001
        self.swing_acc_max = 2.0
        self.pull_acc_max = 1.0

        # load config
        self.robot_to_origin_trans = config['robot_to_origin']['translation']
        self.robot_to_origin_rpy = config['robot_to_origin']['rotation']
        self.picker_to_ee_trans = config['picker_to_ee']['translation']
        self.picker_to_ee_rpy = config['picker_to_ee']['rotation']

        self.matrix_origin2ee_actions = self.get_matrix_transform_origin2base_actions()

    def prepare_traj(self, actions, dt):

        ratio = int(dt / self.dt)

        actions_in_ee = np.array(actions).dot(self.matrix_origin2ee_actions.T)

        goal = FollowCartesianTrajectoryGoal()
        _current_positon = self.get_ee_pose()[0]
        time_from_start = 0

        for i in range(actions_in_ee.shape[0]):
            for j in range(ratio):
                time_from_start += self.dt
                _current_positon += actions_in_ee[i] / ratio
                # Create the Pose message
                pose_msg = geometry_msgs.Pose(
                    geometry_msgs.Vector3(_current_positon[0], _current_positon[1], _current_positon[2]),
                    geometry_msgs.Quaternion(self.init_pose[3], self.init_pose[4], self.init_pose[5], self.init_pose[6])
                )
                # Create the CartesianTrajectoryPoint
                point = CartesianTrajectoryPoint()
                point.pose = pose_msg
                point.time_from_start = rospy.Duration(time_from_start)

                # Add to the goal
                goal.trajectory.points.append(point)

        return goal

    # input actions for TCP
    # output waypoints for EE
    def prepare_stretch_traj(self, actions, dt):
        picker_offset = self.picker_to_ee_trans[2]
        if self.robot_name == "ur10e":
            # left
            tcp_to_ee_trans = np.array([0.0, -picker_offset, 0.0])
        else:
            # right
            tcp_to_ee_trans = np.array([+picker_offset, 0.0, 0.0])
        num_waypts = actions.shape[0]
        goal = FollowCartesianTrajectoryGoal()
        time_from_start = 0
        for i in range(num_waypts):
            time_from_start = i*dt

            cur_TCP_POSI = actions[i]
            cur_EE_POSI  = cur_TCP_POSI + tcp_to_ee_trans
            cur_EE_posi  = self.transform_origin2base(cur_EE_POSI)

            # Create the Pose message
            pose_msg = geometry_msgs.Pose(
                geometry_msgs.Vector3(cur_EE_posi[0], cur_EE_posi[1], cur_EE_posi[2]),
                geometry_msgs.Quaternion(self.init_pose[3], self.init_pose[4], self.init_pose[5], self.init_pose[6])
            )
            # Create the CartesianTrajectoryPoint
            point = CartesianTrajectoryPoint()
            point.pose = pose_msg
            point.time_from_start = rospy.Duration(time_from_start)

            # Add to the goal
            goal.trajectory.points.append(point)
        return goal



    def prepare_move(self, pose, dt=5):
        goal_position = self.transform_origin2base(pose) + self.picker_to_ee_trans
        goal = FollowCartesianTrajectoryGoal()

        # Create initial pose
        point = CartesianTrajectoryPoint()
        point.pose = geometry_msgs.Pose(
            geometry_msgs.Vector3(goal_position[0], goal_position[1], goal_position[2]),
            geometry_msgs.Quaternion(self.init_pose[3], self.init_pose[4], self.init_pose[5], self.init_pose[6])
        )
        point.time_from_start = rospy.Duration(dt)
        goal.trajectory.points.append(point)

        return goal

    def prepare_L_arm_ee_move(self, dist, dt=10):
        goal_EE_posi = 0.001*np.array([670-165-dist*1000,-300,900])
        goal_EE_quat = euler2quat(np.pi / 2, np.pi / 2, np.pi / 2)

        goal = FollowCartesianTrajectoryGoal()
        # Create the Pose message
        pose_msg = geometry_msgs.Pose(
            geometry_msgs.Vector3(goal_EE_posi[0], goal_EE_posi[1], goal_EE_posi[2]),
            geometry_msgs.Quaternion(goal_EE_quat[0], goal_EE_quat[1], goal_EE_quat[2],goal_EE_quat[3])
        )
        # Create the CartesianTrajectoryPoint
        point = CartesianTrajectoryPoint()
        point.pose = pose_msg
        point.time_from_start = rospy.Duration(dt)

        # Add to the goal
        goal.trajectory.points.append(point)

        return goal

    def prepare_R_arm_ee_move(self, dist, dt=5):
        goal_EE_posi = 0.001*np.array([350-dist*1000,300,900])
        # goal_EE_quat = euler2quat(np.pi / 2, np.pi / 2, np.pi / 2)

        goal = FollowCartesianTrajectoryGoal()
        # Create the Pose message
        pose_msg = geometry_msgs.Pose(
            geometry_msgs.Vector3(goal_EE_posi[0], goal_EE_posi[1], goal_EE_posi[2]),
            geometry_msgs.Quaternion(self.init_pose[3], self.init_pose[4], self.init_pose[5], self.init_pose[6])
        )
        # Create the CartesianTrajectoryPoint
        point = CartesianTrajectoryPoint()
        point.pose = pose_msg
        point.time_from_start = rospy.Duration(dt)

        # Add to the goal
        goal.trajectory.points.append(point)

        return goal

    def prepare_tcp_move(self, pose, dt=5):
        picker_offset = self.picker_to_ee_trans[2]
        if self.robot_name == "ur10e":
            # left
            tcp_to_ee_trans = np.array([0.0, -picker_offset, 0.0])
        else:
            # right
            tcp_to_ee_trans = np.array([+picker_offset, 0.0, 0.0])

        cur_TCP_POSI = pose
        cur_EE_POSI = cur_TCP_POSI + tcp_to_ee_trans
        cur_EE_posi = self.transform_origin2base(cur_EE_POSI)

        goal = FollowCartesianTrajectoryGoal()
        # Create the Pose message
        pose_msg = geometry_msgs.Pose(
            geometry_msgs.Vector3(cur_EE_posi[0], cur_EE_posi[1], cur_EE_posi[2]),
            geometry_msgs.Quaternion(self.init_pose[3], self.init_pose[4], self.init_pose[5], self.init_pose[6])
        )
        # Create the CartesianTrajectoryPoint
        point = CartesianTrajectoryPoint()
        point.pose = pose_msg
        point.time_from_start = rospy.Duration(dt)

        # Add to the goal
        goal.trajectory.points.append(point)

        return goal



    def prepare_move_grasp(self, goal_POSI, dt=10):
        if len(goal_POSI) != 3:
            raise ValueError("Position should be a 3D vector")

        goal = FollowCartesianTrajectoryGoal()

        EE_POSI = goal_POSI+np.array([self.picker_to_ee_trans[2], 0, 0])
        offset_piker = 0.025 # offset to make sure the gripper can grab the object
        grasp_POSI = EE_POSI - np.array([offset_piker, 0, 0]) # -offset in x direction
        goal_position = self.transform_origin2base(grasp_POSI)
        point = CartesianTrajectoryPoint()
        point.pose = geometry_msgs.Pose(
            geometry_msgs.Vector3(goal_position[0], goal_position[1], goal_position[2]),
            geometry_msgs.Quaternion(self.init_pose[3], self.init_pose[4], self.init_pose[5], self.init_pose[6])
        )
        point.time_from_start = rospy.Duration(dt)
        goal.trajectory.points.append(point)

        return goal

    def prepare_move_before_grasp(self, goal_POSI, dt=10):
        if len(goal_POSI) != 3:
            raise ValueError("Position should be a 3D vector")

        goal = FollowCartesianTrajectoryGoal()

        midpt_POSI = np.array([self.get_ee_pose_in_origin()[0][0], goal_POSI[1], goal_POSI[2]])
        goal_position = self.transform_origin2base(midpt_POSI)
        point = CartesianTrajectoryPoint()
        point.pose = geometry_msgs.Pose(
            geometry_msgs.Vector3(goal_position[0], goal_position[1], goal_position[2]),
            geometry_msgs.Quaternion(self.init_pose[3], self.init_pose[4], self.init_pose[5], self.init_pose[6])
        )
        point.time_from_start = rospy.Duration(dt)
        goal.trajectory.points.append(point)

        return goal

    def convert2pose_base(self, POSI, QUAT, dt=10):
        posi = self.transform_origin2base(POSI)
        point = CartesianTrajectoryPoint()
        point.pose = geometry_msgs.Pose(
            geometry_msgs.Vector3(posi[0], posi[1], posi[2]),
            geometry_msgs.Quaternion(QUAT[0], QUAT[1], QUAT[2], QUAT[3])
        )
        point.time_from_start = dt
        return point

    def prepare_move_grasp_steps(self, goal_POSI, dt=10):
        delta_y = 0.1
        if len(goal_POSI) != 3:
            raise ValueError("Position should be a 3D vector")

        offset_piker = self.picker_to_ee_trans[2]

        goals = FollowCartesianTrajectoryGoal()
        # middle pose (base frame/ robot frame)
        ## pt 1
        midpt1_quat = euler2quat(np.pi/2, 0, 0)
        midpt1_POSI = self.get_ee_pose_in_origin()[0] + np.array([0.05, 0, -offset_piker])
        point1 = self.convert2pose_base(midpt1_POSI, midpt1_quat, 5)
        goals.trajectory.points.append(point1)

        ## pt 2
        midpt2_quat = midpt1_quat
        midpt2_POSI = midpt1_POSI + np.array([0, goal_POSI[1], goal_POSI[2]])
        point2 = self.convert2pose_base(midpt2_POSI, midpt2_quat, 5)
        goals.trajectory.points.append(point2)

        ## pt 3: grasp pose
        grasp_quat = midpt2_quat
        grasp_POSI = midpt2_POSI + np.array([goal_POSI[1]-0.05, 0, 0])
        point3 = self.convert2pose_base(grasp_POSI, grasp_quat, 5)
        goals.trajectory.points.append(point3)

        return goals

    def send_traj(self, goal, callback=None, wait_result=True):

        if wait_result:
            self.send_cartesian_trajectory(goal, callback)
        else:
            if self.moving is not True:
                threading.Thread(target=self.send_cartesian_trajectory, args=(goal, callback)).start()

    def move_to_init_pose(self, dt=5):

        goal = FollowCartesianTrajectoryGoal()

        # Create initial pose
        point = CartesianTrajectoryPoint()
        point.pose = geometry_msgs.Pose(
            geometry_msgs.Vector3(self.init_pose[0], self.init_pose[1], self.init_pose[2]),
            geometry_msgs.Quaternion(self.init_pose[3], self.init_pose[4], self.init_pose[5], self.init_pose[6])
        )
        point.time_from_start = rospy.Duration(dt)
        goal.trajectory.points.append(point)
        self.send_traj(goal,wait_result=False)
        # self.trajectory_client.send_goal(goal)
        # self.trajectory_client.wait_for_result()
        #
        # result = self.trajectory_client.get_result()
        #
        # rospy.loginfo("Initialization execution finished in state {}".format(result.error_code))

    def send_cartesian_trajectory(self, goal, callback):
        self.moving = True

        self.trajectory_client.send_goal(goal)
        self.trajectory_client.wait_for_result()

        result = self.trajectory_client.get_result()
        self.moving = False
        rospy.loginfo("Trajectory execution finished in state {}".format(result.error_code))
        if callback:
            callback(result)

    def set_gripper_open(self, open):
        """
        Publishes a Bool message to the /set_gripper_open topic
        to open or close the Robotiq 2F85 gripper.
        """

        # Create the Bool message with the desired state
        msg = Bool()
        msg.data = open

        # Log the action
        rospy.loginfo("Gripper open: %s" % open)

        # Publish the message
        self.pub.publish(msg)

        time.sleep(1)

    def done_callback(self, result):
        rospy.signal_shutdown("Task Done")

    def switch_controller(self, target_controller):
        """Activates the desired controller and stops all others from the predefined list above"""
        other_controllers = (
                JOINT_TRAJECTORY_CONTROLLERS
                + CARTESIAN_TRAJECTORY_CONTROLLERS
                + CONFLICTING_CONTROLLERS
        )

        other_controllers.remove(target_controller)

        srv = ListControllersRequest()
        response = self.list_srv(srv)
        for controller in response.controller:
            if controller.name == target_controller and controller.state == "running":
                return

        srv = LoadControllerRequest()
        srv.name = target_controller
        self.load_srv(srv)

        srv = SwitchControllerRequest()
        srv.stop_controllers = other_controllers
        srv.start_controllers = [target_controller]
        srv.strictness = SwitchControllerRequest.BEST_EFFORT
        self.switch_srv(srv)

    def get_ee_pose(self):
        target_frame = '{}_tool0_controller'.format(self.robot_name)
        source_frame = '{}_base'.format(self.robot_name)
        try:
            self.tf_listener.waitForTransform(source_frame, target_frame, rospy.Time(0), rospy.Duration(0))
            tr_target2source = self.tf_listener.lookupTransform(source_frame, target_frame, rospy.Time(0))
            # tf_target2source = self.tf_listener.fromTranslationRotation(*tr_target2source)
            ee_trans = tr_target2source[0]
            ee_quat = tr_target2source[1]
            return ee_trans, ee_quat
        except Exception as e:
            print(e)

    def transform_ee2base(self, pose):
        if len(pose.shape) ==1:
            pose = pose.reshape(1 ,-1)
        ones = np.ones((pose.shape[0], 1))
        pose = np.concatenate((pose, ones), axis=1)

        ee_trans, ee_quat = self.get_ee_pose()
        ee_rpy = quat2euler(ee_quat)
        ee2base = compute_transformation_matrix(ee_trans, ee_rpy)
        pose_in_base = pose.dot(ee2base.T)

        return pose_in_base[:,:3]
    def transform_base2origin(self, pose):
        if len(pose.shape) ==1:
            pose = pose.reshape(1 ,-1)
        ones = np.ones((pose.shape[0], 1))
        pose = np.concatenate((pose, ones), axis=1)

        base2origin = compute_transformation_matrix(self.robot_to_origin_trans, self.robot_to_origin_rpy)
        pose_in_origin = pose.dot(base2origin.T)

        return pose_in_origin[:,:3]

    def transform_origin2base(self, pose):
        if len(pose.shape) ==1:
            pose = pose.reshape(1 ,-1)
        ones = np.ones((pose.shape[0], 1))
        pose = np.concatenate((pose, ones), axis=1)

        base2origin = compute_transformation_matrix(self.robot_to_origin_trans, self.robot_to_origin_rpy)
        origin2base = np.linalg.inv(base2origin)
        pose_in_origin = pose.dot(origin2base.T)

        if pose_in_origin.shape[0] == 1:
            return pose_in_origin[0,:3]
        else:
            return pose_in_origin[:,:3]

    def transform_ee2origin(self, pose):

        return self.transform_base2origin(self.transform_ee2base(pose))

    def get_matrix_transform_origin2base_actions(self):

        base2origin = compute_transformation_matrix(self.robot_to_origin_trans, self.robot_to_origin_rpy)
        origin2base = np.linalg.inv(base2origin)
        # for actions, only orientation is needed
        origin2base[:3, 3] = 0

        return origin2base[:3, :3]

    def get_picker_pose_in_origin(self):
        picker_pose = [self.picker_to_ee_trans]
        picker_pose_in_origin = self.transform_ee2origin(np.array(picker_pose))

        return np.array(picker_pose_in_origin)[0]

    def get_ee_pose_in_origin(self):
        ee_trans, ee_quat = self.get_ee_pose()
        ee_pose_in_origin = self.transform_base2origin(np.array(ee_trans))
        return ee_pose_in_origin

    def run(self):
        # for debug
        actions = np.array([[0.1, 0, 0], [0.1, 0, 0], [0.1, 0, 0]])
        traj = self.prepare_traj(actions, 1.0)
        self.send_traj(traj)

    def _collect_trajectory(self, current_picker_position, target_picker_position):
        """ Policy for collecting data - random sampling"""
        xy_trans = np.random.uniform(0.1, 0.3)
        z_ratio = np.random.uniform(0.1, 0.5)

        norm_direction = np.array([target_picker_position[1, 1] - target_picker_position[0, 1],
                                   target_picker_position[0, 0] - target_picker_position[1, 0]]) / \
                         np.linalg.norm(np.array([target_picker_position[1, 1] - target_picker_position[0, 1],
                                                  target_picker_position[0, 0] - target_picker_position[1, 0]]))

        middle_state = target_picker_position.copy()
        middle_state[:, [0, 1]] = target_picker_position[:, [0, 1]] + xy_trans * norm_direction
        middle_state[:, 2] = current_picker_position[:, 2] + z_ratio * (
                    target_picker_position[:, 2] - current_picker_position[:, 2])

        trajectory_start_to_middle = self._generate_trajectory(current_picker_position, middle_state,
                                                              self.swing_acc_max, self.dt)

        trajectory_middle_to_target = self._generate_trajectory(middle_state, target_picker_position,
                                                            self.pull_acc_max, self.dt)

        trajectory = np.concatenate((trajectory_start_to_middle, trajectory_middle_to_target[1:]), axis=0)
        trajectory = trajectory.reshape(trajectory.shape[0], -1)

        action_list = []
        for step in range(1, trajectory.shape[0]):
            action = np.ones(8, dtype=np.float32)
            action[:3], action[4:7] = trajectory[step, :3] - trajectory[step - 1, :3], trajectory[step,
                                                                                       3:6] - trajectory[step - 1, 3:6]
            action_list.append(action)

        action_list = np.array(action_list)
        action_list[:,[3,7]] = 1
        return action_list


    def _generate_trajectory(self, current_picker_position, target_picker_position, acc_max, dt):
        """ Policy for trajectory generation based on current and target_picker_position"""

        # select column 1 and 2 in current_picker_position and target_picker_position
        initial_vertices_xy = current_picker_position[:, [0, 1]]
        final_vertices_xy = target_picker_position[:, [0, 1]]

        # calculate angle of rotation from initial to final segment in xy plane
        angle = np.arctan2(final_vertices_xy[1, 1] - final_vertices_xy[0, 1],
                           final_vertices_xy[1, 0] - final_vertices_xy[0, 0]) - \
                np.arctan2(initial_vertices_xy[1, 1] - initial_vertices_xy[0, 1],
                           initial_vertices_xy[1, 0] - initial_vertices_xy[0, 0])

        # translation vector: difference between final and initial centers
        translation = (target_picker_position.mean(axis=0) - current_picker_position.mean(axis=0))

        _time_steps = max(np.sqrt(4 * np.abs(translation) / acc_max) / dt)
        steps = np.ceil(_time_steps).max().astype(int)

        # calculate angle of rotation for each step
        rot_steps = angle / steps

        accel_steps = steps // 2
        decel_steps = steps - accel_steps

        v_max = translation * 2 / (steps * dt)
        accelerate = v_max / (accel_steps * dt)
        decelerate = -v_max / (decel_steps * dt)

        # calculate incremental translation
        incremental_translation = [0, 0, 0]

        # initialize list of vertex positions
        positions_xyz = [current_picker_position]


        # apply translation and rotation in each step
        for i in range(steps):
            if i < accel_steps:
                # Acceleration phase
                incremental_translation = (np.divide(incremental_translation,
                                                     self.dt) + accelerate * self.dt) * self.dt
            else:
                # Deceleration phase
                incremental_translation = (np.divide(incremental_translation,
                                                     self.dt) + decelerate * self.dt) * self.dt

            # translate vertices
            vertices = positions_xyz[-1] + incremental_translation

            # calculate rotation matrix for this step
            rotation_matrix = np.array([[np.cos(rot_steps), -np.sin(rot_steps), 0],
                                        [np.sin(rot_steps), np.cos(rot_steps), 0],
                                         [0, 0, 1]])

            # rotate vertices
            center = vertices.mean(axis=0)
            vertices = (rotation_matrix @ (vertices - center).T).T + center

            # append vertices to positions
            positions_xyz.append(vertices)

        return positions_xyz

if __name__ == "__main__":
    rospy.init_node('robot_control', anonymous=True)

    path = os.path.dirname(os.path.abspath(__file__))
    with open(path + '/../cfg/robots.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # init controller client and move to init pose
    robot_left = Robot(config['robot_left'])
    robot_right = Robot(config['robot_right'])

    robot_left.move_to_init_pose()
    robot_right.move_to_init_pose()

    current_picker_position = np.array([robot_left.get_picker_pose_in_origin(), robot_right.get_picker_pose_in_origin()])
    actions = robot_right._collect_trajectory(current_picker_position, np.array([[0.2, -0.2, 0.1], [0.2, 0.2, 0.1]]))
    traj_left = robot_left.prepare_traj(actions[:,:3], robot_right.dt)
    traj_right = robot_right.prepare_traj(actions[:,4:7], robot_right.dt)

    robot_left.send_traj(traj_left)
    robot_right.send_traj(traj_right)
    print(actions)

    # save log to file
    try:
        while not rospy.is_shutdown():
            rospy.sleep(1)
            print("Waiting for rospy shutdown")

        # save log to file
        # np.save('../log/traj_log', pose_cli.pose_log)
        # np.save('../log/traj_desired', client.trajectory_log)

        # print("Saving log to file")

    except KeyboardInterrupt:
        print("Exit")
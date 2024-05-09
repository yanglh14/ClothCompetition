import numpy as np
import random
import pyflex
from softgym.envs.cloth_comp_env import ClothCompEnv
from copy import deepcopy
from softgym.utils.misc import vectorized_range, vectorized_meshgrid
from softgym.utils.pyflex_utils import center_object
from ClothCompetition.utils.camera_utils import get_matrix_world_to_camera, intrinsic_from_fov, get_observable_particle_index

class ClothCompFlattenEnv(ClothCompEnv):
    def __init__(self, cached_states_path='cloth_flatten_init_states.pkl', **kwargs):
        """
        :param cached_states_path:
        :param num_picker: Number of pickers if the aciton_mode is picker
        :param kwargs:
        """
        super().__init__(**kwargs)
        self.get_cached_configs_and_states(cached_states_path, self.num_variations)
        self.prev_covered_area = None  # Should not be used until initialized

    def generate_env_variation(self, num_variations=1, vary_cloth_size=True, vary_cloth_prop=True, render_steps=True):
        """ Generate initial states. Note: This will also change the current states! """
        max_wait_step = 500  # Maximum number of steps waiting for the cloth to stablize
        stable_vel_threshold = 0.01  # Cloth stable when all particles' vel are smaller than this
        generated_configs, generated_states = [], []
        default_config = self.get_default_config()

        for i in range(num_variations):
            self._picker_state = [0, 0]  # Create picker state buffer

            config = deepcopy(default_config)
            self.update_camera(config['camera_name'], config['camera_params'][config['camera_name']])
            if vary_cloth_size:
                cloth_dimx, cloth_dimy = self._sample_cloth_size()
                config['ClothSize'] = [cloth_dimx, cloth_dimy]
            else:
                cloth_dimx, cloth_dimy = config['ClothSize']

            if vary_cloth_prop:
                cloth_prop = self._sample_cloth_prop()
                config['ClothStiff'] = cloth_prop

            # Recalculate the mass of the cloth based on the new cloth size
            dimx, dimy = config['ClothSize']
            num_particles = dimx * dimy
            mass_per_particle = 8e-5
            config['mass'] = num_particles * mass_per_particle
            print("Total mass: ", config['mass'])
            self.set_scene(config)
            self.action_tool.reset([0., -1., 0.])
            pos = pyflex.get_positions().reshape(-1, 4)
            pos[:, :3] -= np.mean(pos, axis=0)[:3]
            pyflex.set_positions(pos.flatten())
            pyflex.set_velocities(np.zeros_like(pos))
            pyflex.step()
            if render_steps:
                pyflex.render()

            ''' select a random pick and lift the fabric '''
            # select a random pick point
            picker_pose, picked_particle_idx = self._random_select_observable_particle_pos(condition="random", offset_direction=[0, 1.0, 0.])
            picked_particles = self._get_picked_particles(right_grasped_particle_idx=picked_particle_idx)
            # Set right arm picker position to this grasp point
            self._set_picker_pos(picker_pose, left=False)
            self._set_picker_pos(np.array([100, 100, 100]), left=True)
            # Let the right arm pick up the heighest particle
            self._move_picker_to_goal(goal_position=np.array([0.0,np.random.random(1) * 0.5 + 0.5, 0.0]), linear_vel=0.2, enable_pick=True,
                                      left=False, picked_particles=picked_particles)

            # Wait for a few seconds for the cloth to be still
            self._wait_until_stable(max_wait_step=200, stable_vel_threshold=0.2, picked_particles=picked_particles)

            ''' Drop the cloth and wait to stablize'''
            self._set_picker_state(0, left=False)
            self._wait_until_stable(max_wait_step=500, stable_vel_threshold=0.2)

            center_object()

            '''The right arm will pick up the heighest particle, and lift it to position [0.0, 0.9, 0.0]'''
            heighest_particle_pos, picked_particle_idx = self._random_select_observable_particle_pos(num_candidates=3, condition="highest",
                                                                                offset_direction=[0., 1.0, 0.])
            picked_particles = self._get_picked_particles(right_grasped_particle_idx=picked_particle_idx)
            # Set right arm picker position to this heighest point
            self._set_picker_pos(heighest_particle_pos, left=False)

            # Let the right arm pick up the heighest particle
            self._move_picker_to_goal(goal_position=np.array([0.0, 0.9, 0.0]), linear_vel=0.2, enable_pick=True,
                                      left=False, picked_particles=picked_particles)

            # Wait for a few seconds for the cloth to be still
            self._wait_until_stable(max_wait_step=200, stable_vel_threshold=0.2, picked_particles=picked_particles)

            '''The left arm will grasp the lowest particle, and lift it up to [0.0, 0.9, 0.0]'''
            # Randomly select the lowest particle
            lowest_particle_pos, picked_particle_idx = self._random_select_observable_particle_pos(num_candidates=3, condition="lowest",
                                                                              offset_direction=[1., 0., 0.])
            picked_particles = self._get_picked_particles(left_grasped_particle_idx=picked_particle_idx)

            # Set left arm picker position to this lowest point
            self._set_picker_pos(lowest_particle_pos, left=True)
            self._set_picker_state(picking = 1, left=True)
            self._set_picker_state(picking = 0, left=False)
            self._set_picker_pos(np.array([0,0.9,-0.3]), left=False)

            # Move the left picker a little bit to the behind
            self._move_picker(delta_position=np.array([0.2, 0.0, 0.0]), enable_pick=True, moving_steps=10, left=True, picked_particles=picked_particles)

            # Wait for a few seconds for the cloth to be still
            self._wait_until_stable(max_wait_step=200, stable_vel_threshold=0.2, picked_particles=picked_particles)

            # Let the left picker go to [0.0, 0.9, 0.0]
            self._move_picker_to_goal(goal_position=np.array([0.0, 0.9, 0.0]), linear_vel=0.2, enable_pick=True,
                                      left=True, picked_particles=picked_particles)
            # Wait for a few seconds for the cloth to be still
            self._wait_until_stable(max_wait_step=500, stable_vel_threshold=0.2, picked_particles=picked_particles)
            config['picked_particles']=picked_particles
            config['picker_pose']=self.action_tool.get_picker_pos()
            if self.action_mode == 'sphere' or self.action_mode.startswith('picker'):
                curr_pos = pyflex.get_positions()
            generated_configs.append(deepcopy(config))
            generated_states.append(deepcopy(self.get_state()))
            self.current_config = config  # Needed in _set_to_flatten function
            generated_configs[-1]['flatten_area'] = self._set_to_flatten()  # Record the maximum flatten area

            print('config {}: camera params {}, flatten area: {}'.format(i, config['camera_params'], generated_configs[-1]['flatten_area']))

        return generated_configs, generated_states


    def _set_to_flatten(self):
        # self._get_current_covered_area(pyflex.get_positions().reshape(-))
        cloth_dimx, cloth_dimz = self.get_current_config()['ClothSize']
        N = cloth_dimx * cloth_dimz
        px = np.linspace(0, cloth_dimx * self.cloth_particle_radius, cloth_dimx)
        py = np.linspace(0, cloth_dimz * self.cloth_particle_radius, cloth_dimz)
        xx, yy = np.meshgrid(px, py)
        new_pos = np.empty(shape=(N, 4), dtype=np.float)
        new_pos[:, 0] = xx.flatten()
        new_pos[:, 1] = self.cloth_particle_radius
        new_pos[:, 2] = yy.flatten()
        new_pos[:, 3] = 1.
        new_pos[:, :3] -= np.mean(new_pos[:, :3], axis=0)
        pyflex.set_positions(new_pos.flatten())
        return self._get_current_covered_area(new_pos, axis_idx =0)

    def _reset(self):
        """ Right now only use one initial state"""
        self.prev_covered_area = self._get_current_covered_area(pyflex.get_positions())
        if hasattr(self, 'action_tool'):
            curr_pos = pyflex.get_positions()
            cx, cy = self._get_center_point(curr_pos)
            self.action_tool.reset([cx, 0.2, cy])
        pyflex.step()
        self.init_covered_area = None
        info = self._get_info()
        self.init_covered_area = info['performance']
        return self._get_obs()

    def _step(self, action, picked_particles=[None, None] ):
        self.action_tool.step(action, picked_particles)
        if self.action_mode in ['sawyer', 'franka']:
            pyflex.step(self.action_tool.next_action)
        else:
            pyflex.step()
        return

    def _get_current_covered_area(self, pos, axis_idx=1):
        """
        Calculate the covered area by taking max x,y cood and min x,y coord, create a discritized grid between the points
        :param pos: Current positions of the particle states
        """
        pos = np.reshape(pos, [-1, 4])
        min_x = np.min(pos[:, axis_idx])
        min_y = np.min(pos[:, 2])
        max_x = np.max(pos[:, axis_idx])
        max_y = np.max(pos[:, 2])
        init = np.array([min_x, min_y])
        span = np.array([max_x - min_x, max_y - min_y]) / 100.
        pos2d = pos[:, [axis_idx, 2]]

        offset = pos2d - init
        slotted_x_low = np.maximum(np.round((offset[:, 0] - self.cloth_particle_radius) / span[0]).astype(int), 0)
        slotted_x_high = np.minimum(np.round((offset[:, 0] + self.cloth_particle_radius) / span[0]).astype(int), 100)
        slotted_y_low = np.maximum(np.round((offset[:, 1] - self.cloth_particle_radius) / span[1]).astype(int), 0)
        slotted_y_high = np.minimum(np.round((offset[:, 1] + self.cloth_particle_radius) / span[1]).astype(int), 100)
        # Method 1
        grid = np.zeros(10000)  # Discretization
        listx = vectorized_range(slotted_x_low, slotted_x_high)
        listy = vectorized_range(slotted_y_low, slotted_y_high)
        listxx, listyy = vectorized_meshgrid(listx, listy)
        idx = listxx * 100 + listyy
        idx = np.clip(idx.flatten(), 0, 9999)
        grid[idx] = 1

        return np.sum(grid) * span[0] * span[1]

        # Method 2
        # grid_copy = np.zeros([100, 100])
        # for x_low, x_high, y_low, y_high in zip(slotted_x_low, slotted_x_high, slotted_y_low, slotted_y_high):
        #     grid_copy[x_low:x_high, y_low:y_high] = 1
        # assert np.allclose(grid_copy, grid)
        # return np.sum(grid_copy) * span[0] * span[1]

    def _get_center_point(self, pos):
        pos = np.reshape(pos, [-1, 4])
        min_x = np.min(pos[:, 0])
        min_y = np.min(pos[:, 2])
        max_x = np.max(pos[:, 0])
        max_y = np.max(pos[:, 2])
        return 0.5 * (min_x + max_x), 0.5 * (min_y + max_y)

    def compute_reward(self, action=None, obs=None, set_prev_reward=False):
        particle_pos = pyflex.get_positions()
        curr_covered_area = self._get_current_covered_area(particle_pos)
        r = curr_covered_area
        return r

    # @property
    # def performance_bound(self):
    #     dimx, dimy = self.current_config['ClothSize']
    #     max_area = dimx * self.cloth_particle_radius * dimy * self.cloth_particle_radius
    #     min_p = 0
    #     max_p = max_area
    #     return min_p, max_p

    def _get_info(self):
        # Duplicate of the compute reward function!
        particle_pos = pyflex.get_positions()
        curr_covered_area = self._get_current_covered_area(particle_pos)
        if hasattr(self, 'init_covered_area'):
            init_covered_area = curr_covered_area if self.init_covered_area is None else self.init_covered_area
        else:
            init_covered_area = curr_covered_area
        current_config = self.get_current_config()
        if 'flatten_area' in current_config:
            max_covered_area = current_config['flatten_area']
        else:
            max_covered_area = curr_covered_area
        info = {
            'performance': curr_covered_area,
            'normalized_performance': (curr_covered_area - init_covered_area) / (max_covered_area - init_covered_area),
            'normalized_performance_2': (curr_covered_area) / (max_covered_area)
        }
        if 'qpg' in self.action_mode:
            info['total_steps'] = self.action_tool.total_steps
        return info

    def get_picked_particle(self):
        pps = np.ones(shape=self.action_tool.num_picker, dtype=np.int32)  * -1 # -1 means no particles picked
        for i, pp in enumerate(self.action_tool.picked_particles):
            if pp is not None:
                pps[i] = pp
        return pps

    @property
    def _left_picker_index(self):
        return 1

    @property
    def _right_picker_index(self):
        return 0

    def _move_picker_to_goal(self, goal_position, linear_vel=0.2, enable_pick=True, left=False, picked_particles=[None, None]):
        '''Move the picker to the goal position linearly with a constant speed

        Args:
            goal_position (np.array): 3D position of the goal
            linear_vel (float): Linear velocity of the picker
            left (bool): Whether the picker is the left picker (0 for right picker, 1 for left picker)
        '''
        delta_action_per_step, steps_to_goal = self._plan_linear_picker_motion(goal_position, linear_vel, left)

        for _ in range(0, steps_to_goal + 1):
            # Move the picker to the goal position linearly with a constant speed
            action = self._compose_picker_action(delta_action_per_step, enable_pick=enable_pick, left=left)
            self.step(action, picked_particles=picked_particles)
        # Set the picker to goal position to ensure the picker reached the goal position
        self._set_picker_pos(goal_position, left)
        return picked_particles

    def _get_picked_particles(self, right_grasped_particle_idx=None, left_grasped_particle_idx=None):
        picked_particles = [None, None]
        if right_grasped_particle_idx is not None:
            picked_particles[self._picker_index(False)] = right_grasped_particle_idx
        if left_grasped_particle_idx is not None:
            picked_particles[self._picker_index(True)] = left_grasped_particle_idx
        return picked_particles

    def _plan_linear_picker_motion(self, goal_position, linear_vel=0.2, left=False):
        '''Calculate delta linear motion for the picker to reach the goal position

        Args:
            goal_position (np.array): 3D position of the goal
            linear_vel (float): Linear velocity of the picker
            left (bool): Whether the picker is the left picker (0 for right picker, 1 for left picker)
        '''
        current_picker_pos, _ = self.action_tool._get_pos()
        # Caculate the moving direction
        goal_error_position = goal_position - current_picker_pos[self._picker_index(left)]
        goal_error_distance = np.linalg.norm(goal_error_position)
        moving_direction = goal_error_position / goal_error_distance
        # Calculate the moving distance
        moving_distance_per_step = linear_vel * 0.01
        # Calculate delta action per step (delta position)
        delta_action_per_step = moving_distance_per_step * moving_direction
        # Calculate the steps needed to reach goal
        steps_to_goal = int(goal_error_distance / moving_distance_per_step)

        return delta_action_per_step, steps_to_goal

    def _compose_picker_action(self, delta_action, enable_pick=False, left=False):
        action = np.zeros_like(self.action_space.sample())
        if left:
            action[4:7] = delta_action
            if enable_pick:
                action[7] = 1
                self._set_picker_state(1, left=True)
            else:
                self._set_picker_state(0, left=True)

            # For another picker, keep it as it is
            action[3] = self._get_picker_state(left=False)
        else:
            action[:3] = delta_action
            if enable_pick:
                action[3] = 1
                self._set_picker_state(1, left=False)
            else:
                self._set_picker_state(0, left=False)

            # For another picker, keep it as it is
            action[7] = self._get_picker_state(left=True)

        return action

    def _set_picker_state(self, picking, left=False):
        '''set picker state (picking or unpicking)'''
        if left:
            self._picker_state[self._left_picker_index] = picking
        else:
            self._picker_state[self._right_picker_index] = picking

    def _get_picker_state(self, left=False):
        return self._picker_state[self._left_picker_index] if left else self._picker_state[self._right_picker_index]

    def _picker_index(self, left=False):
        if left:
            return self._left_picker_index
        else:
            return self._right_picker_index

    def _set_picker_pos(self, position, left=False):
        '''Set picker to position for a specific picker (left or right)

        Args:
            position (np.array): 3D position of the picker
            left (bool): Whether the picker is the left picker (0 for right picker, 1 for left picker)
        '''
        current_picker_pos, _ = self.action_tool._get_pos()
        current_picker_pos[self._picker_index(left)] = position
        self.action_tool.set_picker_pos(current_picker_pos)

    def _random_select_observable_particle_pos(self, num_candidates=3, condition: str = "lowest",
                                               offset_direction=[0., 1.0, 0.]):
        '''Randomly select observable pickles based on the condition

        Args:
            num_candidates (int): Number of candidates to select, only applicable when condition is "lowest" or "highest"
            condition (str): Condition to select the pickles, available options are "lowest", "highest", and "random"
            offset_direction (np.array): Offset direction for the picker to pick the particle
        '''
        assert condition in ["lowest", "highest", "random"], "Invalid condition"

        # Get observable particle indices
        curr_data = self.get_curr_env_data()
        observable_idx = curr_data['observable_idx']

        if condition == "lowest" or condition == "highest":
            observable_heights = curr_data['positions'][observable_idx, 1]
            if condition == "lowest":
                candidates = observable_idx[
                    np.argpartition(observable_heights, num_candidates, axis=0)[:num_candidates]]
            else:
                candidates = observable_idx[np.argpartition(observable_heights, -num_candidates)[-num_candidates:]]
            # Randomly select one candidate
            rand_choise = np.random.choice(candidates)
        else:
            # Just randomly select from all observable indices
            rand_choise = np.random.choice(observable_idx)

        picker_offset = self.picker_radius + self.cloth_particle_radius
        return curr_data['positions'][rand_choise] + np.array(offset_direction) * picker_offset, rand_choise


    def get_curr_env_data(self):
        # Env info that does not change within one episode
        config = self.get_current_config()
        cloth_xdim, cloth_ydim = config['ClothSize']

        position = pyflex.get_positions().reshape(-1, 4)[:, :3]
        picker_position = self.action_tool.get_picker_pos()

        # Cloth and picker information
        # Get partially observed particle index
        rgbd = self.get_rgbd(show_picker=False)
        rgb, depth = rgbd[:, :, :3], rgbd[:, :, 3]

        world_coordinates = self.get_world_coords(rgb, depth)

        observable_idx = get_observable_particle_index(world_coordinates, position, rgb, depth)

        ret = {'positions': position.astype(np.float32),
               'picker_position': picker_position,
               'observable_idx': observable_idx,}
        return ret

    def get_world_coords(self, rgb, depth):
        height, width, _ = rgb.shape
        K = intrinsic_from_fov(height, width, 45)  # the fov is 90 degrees

        # Apply back-projection: K_inv @ pixels * depth
        u0 = K[0, 2]
        v0 = K[1, 2]
        fx = K[0, 0]
        fy = K[1, 1]

        x = np.linspace(0, width - 1, width).astype(np.float)
        y = np.linspace(0, height - 1, height).astype(np.float)
        u, v = np.meshgrid(x, y)
        one = np.ones((height, width, 1))
        x = (u - u0) * depth / fx
        y = (v - v0) * depth / fy
        z = depth
        cam_coords = np.dstack([x, y, z, one])

        matrix_world_to_camera = get_matrix_world_to_camera(
            self.camera_params[self.camera_name]['pos'], self.camera_params[self.camera_name]['angle'])

        # convert the camera coordinate back to the world coordinate using the rotation and translation matrix
        cam_coords = cam_coords.reshape((-1, 4)).transpose()  # 4 x (height x width)
        world_coords = np.linalg.inv(matrix_world_to_camera) @ cam_coords  # 4 x (height x width)
        world_coords = world_coords.transpose().reshape((height, width, 4))

        return world_coords

    def _move_picker(self, delta_position, enable_pick=True, moving_steps=50, left=False , picked_particles=[None,None]):
        '''Move the picker by delta position at a constant speed

        Args:
            delta_position (np.array): 3D position of the goal
            enable_pick (bool): Whether the picker is in picking mode
            moving_period (int): Number of steps to move the picker
            left (bool): Whether the picker is the left picker (0 for right picker, 1 for left picker)
        Be careful when you want to release a picker while another picker is grabbing a particle! only eff
        '''
        delta_position_per_step = delta_position / moving_steps
        for _ in range(moving_steps):
            # Move the picker to the goal position
            action = self._compose_picker_action(delta_position_per_step, enable_pick=enable_pick, left=left)
            self.step(action, picked_particles=picked_particles)

    def _wait_until_stable(self, max_wait_step=100, stable_vel_threshold=1e-3, picked_particles=[None, None]):
        '''Wait until the cloth is stable

        Args:
            max_wait_step (int): Maximum number of steps to wait
            stable_vel_threshold (float): Velocity threshold to determine if the cloth is stable
        '''
        for j in range(0, max_wait_step):
            curr_vel = pyflex.get_velocities()
            action = np.zeros_like(self.action_space.sample())
            # Keep the picker state as it is
            action[3] = self._get_picker_state(left=False)
            action[7] = self._get_picker_state(left=True)
            self.step(action, picked_particles = picked_particles)
            if np.alltrue(np.abs(curr_vel) < stable_vel_threshold) and j > 5:
                break
        print("[Warning] Cloth is not stable after {} steps".format(max_wait_step))

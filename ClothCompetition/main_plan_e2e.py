import numpy as np
from ClothCompetition.rs_planner import RandomShootingUVPickandPlacePlanner
from chester import logger
import json
import os.path as osp

import copy
import pyflex
import pickle
import multiprocessing as mp
from ClothCompetition.utils.utils import (
    downsample, transform_info, draw_planned_actions, visualize, draw_edge,
    pc_reward_model, voxelize_pointcloud, vv_to_args, set_picker_pos, cem_make_gif, configure_seed, configure_logger
)
from ClothCompetition.utils.camera_utils import get_matrix_world_to_camera, get_world_coords
from softgym.utils.visualization import save_numpy_as_gif

from ClothCompetition.vc_dynamics_e2e import VCDynamics
from ClothCompetition.vc_edge import VCConnection
import argparse
import torch


def get_default_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='release', help='Name of the experiment')
    parser.add_argument('--log_dir', type=str, default='data/plan/', help='Logging directory')
    parser.add_argument('--seed', type=int, default=100)

    # Env
    parser.add_argument('--env_name', type=str, default='ClothFlatten', help="'ClothFlatten' or 'TshirtFlatten'")
    parser.add_argument('--cloth_type', type=str, default='tshirt-small', help="For 'TshirtFlatten', what types of tshir to use")
    parser.add_argument('--cached_states_path', type=str, default='cloth_flatten_init_states_test_40_2.pkl') 
    parser.add_argument('--num_variations', type=int, default=20) 
    parser.add_argument('--camera_name', type=str, default='default_camera')
    parser.add_argument('--down_sample_scale', type=int, default=3)
    parser.add_argument('--n_his', type=int, default=5)

    # Load model
    parser.add_argument('--edge_model_path', type=str, default=None,
                        help='Path to a trained edgeGNN model')
    parser.add_argument('--partial_dyn_path', type=str, default=None,
                        help='Path to a dynamics model using partial point cloud')
    parser.add_argument('--load_optim', type=bool, default=False, help='Load optimizer when resume training')

    # Planning
    parser.add_argument('--shooting_number', type=int, default=500, help='Number of sampled pick-and-place action for random shooting')
    parser.add_argument('--delta_y', type=float, default=0.07, help='Fixed picking height for real-world experiment')
    parser.add_argument('--delta_y_range', type=list, default=[0, 0.5], help='Sample range for the pick-and-place height in simulation')
    parser.add_argument('--move_distance_range', type=list, default=[0.05, 0.2], help='Sample range for the pick-and-place distance')
    parser.add_argument('--pull_step', type=int, default=10, help='Number of steps for doing pick-and-place on the cloth')
    parser.add_argument('--wait_step', type=int, default=6, help='Number of steps for waiting the cloth to stablize after the pick-and-place')
    parser.add_argument('--num_worker', type=int, default=6, help='Number of processes to generate the sampled pick-and-place actions in parallel')
    parser.add_argument('--task', type=str, default='flatten', help="'flatten' or 'fold'")
    parser.add_argument('--pred_time_interval', type=int, default=5, help='Interval of timesteps between each dynamics prediction (model dt)')
    parser.add_argument('--configurations', type=list, default=[i for i in range(20)], help='List of configurations to run')
    parser.add_argument('--pick_and_place_num', type=int, default=10, help='Number of pick-and-place for one smoothing trajectory')

    # Other
    parser.add_argument('--cuda_idx', type=int, default=0)
    parser.add_argument('--voxel_size', type=float, default=0.0216, help='Pointcloud voxelization size')
    parser.add_argument('--sensor_noise', type=float, default=0, help='Artificial noise added to depth sensor')
    parser.add_argument('--gpu_num', type=int, default=1, help='# of GPUs to be used')

    # Ablation
    parser.add_argument('--fix_collision_edge', type=int, default=0, help="""
        for ablation that train without mesh edges, 
        if True, fix collision edges from the first time step during planning; 
        If False, recompute collision edge at each time step
    """)
    parser.add_argument('--use_collision_as_mesh_edge', type=int, default=0, help="""
        for ablation that train with mesh edges, but remove edge GNN at test time, 
        so it uses first-time step collision edges as the mesh edges
    """)

    args = parser.parse_args()
    return args


def prepare_policy():
    # move one of the picker to be under ground
    shape_states = pyflex.get_shape_states().reshape(-1, 14)
    shape_states[1, :3] = -1
    shape_states[1, 3:6] = -1

    # move another picker to be above the cloth
    pos = pyflex.get_positions().reshape((-1, 4))[:, :3]
    pp = np.random.randint(len(pos))
    shape_states[0, :3] = pos[pp] + [0., 0.06, 0.]
    shape_states[0, 3:6] = pos[pp] + [0., 0.06, 0.]
    pyflex.set_shape_states(shape_states.flatten())


def create_env(args):
    from softgym.registered_env import env_arg_dict
    from softgym.registered_env import SOFTGYM_ENVS

    # create env
    env_args = copy.deepcopy(env_arg_dict[args.env_name])
    env_args['render_mode'] = 'both'
    env_args['observation_mode'] = 'cam_rgb'
    env_args['render'] = True
    env_args['camera_height'] = 360
    env_args['camera_width'] = 360
    env_args['camera_name'] = args.camera_name
    env_args['headless'] = True
    env_args['action_repeat'] = 1
    env_args['picker_radius'] = 0.01
    env_args['picker_threshold'] = 0.00625
    assert args.env_name in ['ClothFlatten', 'TshirtFlatten']
    env_args['cached_states_path'] = args.cached_states_path
    env_args['num_variations'] = args.num_variations
    if args.env_name == 'TshirtFlatten':
        env_args['cloth_type'] = args.cloth_type

    env = SOFTGYM_ENVS[args.env_name](**env_args)
    render_env_kwargs = copy.deepcopy(env_args)
    render_env_kwargs['render_mode'] = 'particle'
    render_env = SOFTGYM_ENVS[args.env_name](**render_env_kwargs)

    return env, render_env


def load_edge_model(edge_model_path, env):
    if edge_model_path is not None:
        edge_model_dir = osp.dirname(edge_model_path)
        edge_model_vv = json.load(open(osp.join(edge_model_dir, 'best_state.json')))
        edge_model_vv['eval'] = 1
        edge_model_vv['n_epoch'] = 1
        edge_model_vv['edge_model_path'] = edge_model_path
        edge_model_args = vv_to_args(edge_model_vv)

        vcd_edge = VCConnection(edge_model_args, env=env)
        print('edge GNN model successfully loaded from ', edge_model_path, flush=True)
    else:
        print("no edge GNN model is loaded")
        vcd_edge = None

    return vcd_edge


def load_dynamics_model(args, env, vcd_edge):
    model_vv_dir = osp.dirname(args.partial_dyn_path)
    model_vv = json.load(open(osp.join(model_vv_dir, 'best_state.json')))

    model_vv[
        'fix_collision_edge'] = args.fix_collision_edge  # for ablation that train without mesh edges, if True, fix collision edges from the first time step during planning; If False, recompute collision edge at each time step
    model_vv[
        'use_collision_as_mesh_edge'] = args.use_collision_as_mesh_edge  # for ablation that train with mesh edges, but remove edge GNN at test time, so it uses first-time step collision edges as the mesh edges
    model_vv['train_mode'] = 'vsbl'
    model_vv['use_wandb'] = False
    model_vv['eval'] = 1
    model_vv['load_optim'] = False
    model_vv['pred_time_interval'] = args.pred_time_interval
    model_vv['cuda_idx'] = args.cuda_idx
    model_vv['partial_dyn_path'] = args.partial_dyn_path
    args = vv_to_args(model_vv)

    vcdynamics = VCDynamics(args, vcd_edge=vcd_edge, env=env)
    return vcdynamics


def get_rgbd_and_mask(env, sensor_noise):
    rgbd = env.get_rgbd(show_picker=False)
    rgb = rgbd[:, :, :3]
    depth = rgbd[:, :, 3]
    if sensor_noise > 0:
        non_cloth_mask = (depth <= 0)
        depth += np.random.normal(loc=0, scale=sensor_noise,
                                  size=(depth.shape[0], depth.shape[1]))
        depth[non_cloth_mask] = 0

    return depth.copy(), rgb, depth

class Planner(object):
    def __init__(self):
        args = get_default_args()
        mp.set_start_method('forkserver', force=True)

        # Configure logger
        configure_logger(args.log_dir, args.exp_name)
        # Configure seed
        configure_seed(args.seed)
        with open(osp.join(logger.get_dir(), 'variant.json'), 'w') as f:
            json.dump(args.__dict__, f, indent=2, sort_keys=True)

        # create env
        env, render_env = None, None

        # load vcdynamics
        vcd_edge = load_edge_model(args.edge_model_path, env)
        self.vcdynamics = load_dynamics_model(args, env, vcd_edge)
    def inference(self, point_clouds, final_candidates, real_robot=False):
        performances =[]
        for candidate in final_candidates:
            point_clouds_copy = point_clouds.copy()
            performance = self.vcdynamics.infer_performance(point_clouds_copy, candidate, real_robot)
            performances.append(performance)

        # best_pick_points_idx = np.argmax(np.array(performances))
        # list of candidates from the best to the worst
        best_pick_points_idx = np.argsort(np.array(performances).squeeze())[::-1]
        return final_candidates[best_pick_points_idx]
        # return final_candidates[best_pick_points_idx]
def main(args):
    mp.set_start_method('forkserver', force=True)

    # Configure logger
    configure_logger(args.log_dir, args.exp_name)
    # Configure seed
    configure_seed(args.seed)

    # Dump parameters
    with open(osp.join(logger.get_dir(), 'variant.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2, sort_keys=True)

    # create env
    env, render_env = create_env(args)

    # load vcdynamics
    vcd_edge = load_edge_model(args.edge_model_path, env)
    vcdynamics = load_dynamics_model(args, env, vcd_edge)
    vcdynamics.infer_performance()
if __name__ == '__main__':
    main(get_default_args())

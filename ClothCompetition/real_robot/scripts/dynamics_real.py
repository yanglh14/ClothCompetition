import os
import os.path as osp
import copy
import cv2
import json
import wandb
import numpy as np
import scipy
from tqdm import tqdm
from chester import logger
from omegaconf import OmegaConf

import torch
import torch_geometric

from ADFM.module.models import GNN
from ADFM.module.dataset import ClothDataset
from ADFM.utils.data_utils import AggDict
from ADFM.utils.utils import extract_numbers, pc_reward_model, visualize, cloth_drop_reward_fuc, save_numpy_as_gif
from ADFM.utils.camera_utils import get_matrix_world_to_camera, project_to_image

class DynamicReal(object):
    def __init__(self, args, env, edge=None):
        self.args = args
        self.env = env
        self.train_mode = args.train_mode
        self.device = torch.device(self.args.cuda_idx)

        self.input_types = ['full', 'vsbl'] if self.train_mode == 'graph_imit' else [self.train_mode]
        self.output_type = args.output_type

        self.models, self.optims, self.schedulers = {}, {}, {}
        for m in self.input_types:
            self.models[m] = GNN(args, decoder_output_dim=3, name=m, use_reward=False if self.train_mode == 'vsbl' else True)
            lr = getattr(self.args, m + '_lr') if hasattr(self.args, m + '_lr') else self.args.lr
            self.optims[m] = torch.optim.Adam(self.models[m].param(), lr=lr, betas=(self.args.beta1, 0.999))
            self.schedulers[m] = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optims[m], 'min', factor=0.8,
                                                                            patience=3, verbose=True)
            self.models[m].to(self.device)
        self.edge = edge
        self.load_model(self.args.load_optim)
        print("ADFM dynamics models created")

        self.log_dir = logger.get_dir()

    def load_model(self, load_optim=False):
        if self.train_mode == 'vsbl' and self.args.partial_dyn_path is not None:  # Resume training of partial model
            self.models['vsbl'].load_model(self.args.partial_dyn_path, load_optim=load_optim, optim=self.optims['vsbl'])
            self.load_epoch = int(extract_numbers(self.args.partial_dyn_path)[-1])

        if self.train_mode == 'full' and self.args.full_dyn_path is not None:  # Resume training of full model
            self.models['full'].load_model(self.args.full_dyn_path, load_optim=load_optim, optim=self.optims['full'])
            self.load_epoch = int(extract_numbers(self.args.full_dyn_path)[-1])

        if self.train_mode == 'graph_imit' and self.args.full_dyn_path is not None:
            # Imitating the full model using a partial model.
            # Need to first load the full model, and then copy weights to the partial model
            self.models['full'].load_model(self.args.full_dyn_path, load_optim=False)
            self.models['vsbl'].load_model(self.args.full_dyn_path, load_optim=False, load_names=self.args.copy_teach)
            self.load_epoch = 0

    def rollout(self, args):

        model_input_data = args['model_input_data']
        actions = args['actions']  # NOTE: sequence of actions to rollout
        reward_model = args['reward_model']
        m_name = args.get('m_name', 'vsbl')
        H = len(actions)  # Planning horizon
        cuda_idx = args.get('cuda_idx', 0)
        robot_exp = args.get('robot_exp', False)

        self.set_mode('eval')
        self.to(cuda_idx)
        self.device = torch.device(cuda_idx)

        pc_pos = model_input_data['pointcloud']
        pc_vel_his = model_input_data['vel_his']
        picker_pos = model_input_data['picker_position']
        mesh_edges = model_input_data.get('mesh_edges', None)
        target_keypoints = model_input_data['target_keypoints']

        # record model predicted point cloud positions
        model_positions = np.zeros((H, len(pc_pos), 3))
        shape_positions = np.zeros((H, 2, 3))
        initial_pc_pos = pc_pos.copy()
        pred_rewards = np.zeros(H)
        gt_pos_rewards = np.zeros(H)

        # Predict mesh during evaluation and use gt edges during training
        if self.edge is not None and mesh_edges is None:
            model_input_data['cuda_idx'] = cuda_idx
            mesh_edges = self.edge.infer_mesh_edges(model_input_data)

        final_ret = None
        for t in range(H):
            data = {'pointcloud': pc_pos,
                    'vel_his': pc_vel_his,
                    'picker_position': picker_pos,
                    'action': actions[t],
                    'mesh_edges': mesh_edges}
            graph_data = self.build_graph(data, input_type=m_name, robot_exp=robot_exp)

            model_positions[t] = pc_pos
            shape_positions[t] = picker_pos

            inputs = {'x': graph_data['node_attr'].to(self.device),
                      'edge_attr': graph_data['edge_attr'].to(self.device),
                      'edge_index': graph_data['neighbors'].to(self.device),
                      'x_batch': torch.zeros(graph_data['node_attr'].size(0), dtype=torch.long, device=self.device),
                      'u': torch.zeros([1, self.args.global_size], device=self.device)}

            with torch.no_grad():
                pred = self.models[m_name](inputs)
                pred_ = pred['pred'].cpu().numpy()
                pred_reward = pred['reward_nxt'].cpu().numpy() if 'reward_nxt' in pred else 0.

            pc_pos, pc_vel_his, picker_pos = self.update_graph(pred_, pc_pos, pc_vel_his,
                                                               graph_data['picked_status'],
                                                               graph_data['picked_particles'], model_input_data)

            pred_rewards[t] = pred_reward
            gt_pos_rewards[t] = 0

            if t == H - 1:
                reward = reward_model(pc_pos, target_keypoints)

                final_ret = reward

        return dict(final_ret=final_ret,
                    model_positions=model_positions,
                    shape_positions=shape_positions,
                    mesh_edges=mesh_edges,
                    pred_rewards=pred_rewards,
                    gt_pos_rewards=gt_pos_rewards)
    def update_graph(self, pred_, pc_pos, velocity_his, picked_status, picked_particles,model_input_data):
        """ Euler integration"""
        # vel_his: [v(t-20), ... v(t)], v(t) = (p(t) - o(t-5)) / (5*dt)
        pred_time_interval = self.args.pred_time_interval

        if self.output_type == 'accel':
            pred_vel = velocity_his[:, -3:] + pred_ * self.args.dt * pred_time_interval
        elif self.output_type == 'vel':
            pred_vel = pred_
        else:
            raise NotImplementedError

        pc_pos = pc_pos + pred_vel * self.args.dt * pred_time_interval
        pc_pos[:, 1] = np.maximum(pc_pos[:, 1], self.args.particle_radius)  # z should be non-negative

        if self.args.env_shape is not None:
            env_shape = self.args.env_shape
            if env_shape == 'platform':
                pc_pos_ = abs(pc_pos - model_input_data['shape_pos']) - model_input_data['shape_size']
                # check pc_pos_ < 0: if true, then the particle is inside the box, and should be moved up outside
                for i in range(pc_pos_.shape[0]):
                    if pc_pos_[i, 0] < 0 and pc_pos_[i, 1] < 0 and pc_pos_[i, 2] < 0:
                        pc_pos[i, 1] = model_input_data['shape_size'][1] + self.args.particle_radius
            elif env_shape == 'sphere':
                pc_pos_ = np.linalg.norm(pc_pos - model_input_data['shape_pos'], axis=1) - model_input_data['shape_size'][0]
                for i in range(pc_pos_.shape[0]):
                    if pc_pos_[i] < 0:
                        pc_pos[i] = model_input_data['shape_pos'] + (pc_pos[i] - model_input_data['shape_pos']) / np.linalg.norm(pc_pos[i] - model_input_data['shape_pos']) * model_input_data['shape_size'][0]
            elif env_shape == 'rod':
                vector_torod = pc_pos - model_input_data['shape_pos']
                vector_torod[:,2] = 0
                pc_pos_ = np.linalg.norm(vector_torod, axis=1) - model_input_data['shape_size'][0]
                # check pc_pos_ < 0: if true, then the particle is inside the rod, and should be moved up outside
                for i in range(pc_pos_.shape[0]):
                    if pc_pos_[i] < 0:
                        _pc_pos = model_input_data['shape_pos'] + vector_torod[i] / np.linalg.norm(vector_torod[i]) * model_input_data['shape_size'][0]
                        pc_pos[i][:2] = _pc_pos[:2]
        # udpate position and velocity from the model prediction
        velocity_his = np.hstack([velocity_his[:, 3:], pred_vel])

        # the picked particles position and velocity should remain the same as before
        cnt = 0
        picked_vel, picked_pos, new_picker_pos = picked_status
        for p_idx in picked_particles:
            if p_idx != -1:
                pc_pos[p_idx] = picked_pos[cnt]
                velocity_his[p_idx] = picked_vel[cnt]
                cnt += 1

        # update picker position, and the particles picked
        picker_pos = new_picker_pos
        return pc_pos, velocity_his, picker_pos

    def set_mode(self, mode='train'):
        for model in self.models.values():
            model.set_mode('train' if mode == 'train' else 'eval')

    def to(self, cuda_idx):
        for model in self.models.values():
            model.to(torch.device("cuda:{}".format(cuda_idx)))

    def build_graph(self, data, input_type, robot_exp=False):
        """
        data: positions, vel_history, picked_points, picked_point_positions, scene_params
        downsample: whether to downsample the graph
        test: if False, we are in the training mode, where we know exactly the picked point and its movement
            if True, we are in the test mode, we have to infer the picked point in the (downsampled graph) and compute
                its movement.

        return:
        node_attr: N x (vel_history x 3)
        edges: 2 x E, the edges
        edge_attr: E x edge_feature_dim
        global_feat: fixed, not used for now
        """

        vox_pc, velocity_his = data['pointcloud'], data['vel_his']
        picked_points, picked_status = self._find_and_update_picked_point(data, robot_exp=robot_exp)  # Return index of the picked point

        node_attr = self._compute_node_attr(vox_pc, picked_points, velocity_his, **data)

        edges, edge_attr = self._compute_edge_attr(input_type, data)

        return {'node_attr': node_attr,
                'neighbors': edges,
                'edge_attr': edge_attr,
                'picked_particles': picked_points,
                'picked_status': picked_status}

    def _find_and_update_picked_point(self, data, robot_exp):
        """ Directly change the position and velocity of the picked point so that the dynamics model understand the action"""
        picked_pos = []  # Position of the picked particle
        picked_velocity = []  # Velocity of the picked particle

        action = (data['action'] * self.args.action_repeat).reshape([-1, 4])  # scale to the real action

        vox_pc, picker_pos, velocity_his = data['pointcloud'], data['picker_position'], data['vel_his']

        picked_particles = [-1 for _ in picker_pos]
        pick_flag = action[:, 3] > 0.5
        new_picker_pos = picker_pos.copy()
        if robot_exp:

            for i in range(self.args.num_picker):
                new_picker_pos[i, :] = picker_pos[i, :] + action[i, :3]

                if pick_flag[i]:
                    if picked_particles[i] == -1:  # No particle is currently picked and thus need to select a particle to pick
                        dists = scipy.spatial.distance.cdist(picker_pos[i].reshape((-1, 3)), vox_pc[:, :3].reshape((-1, 3)))
                        picked_particles[i] = np.argmin(dists)

                    if picked_particles[i] != -1:
                        old_pos = vox_pc[picked_particles[i]]
                        new_pos = vox_pc[picked_particles[i]] + action[i, :3]
                        new_vel = (new_pos - old_pos) / (self.args.dt*self.args.pred_time_interval)

                        tmp_vel_history = (velocity_his[picked_particles[i]][3:]).copy()
                        velocity_his[picked_particles[i], :-3] = tmp_vel_history
                        velocity_his[picked_particles[i], -3:] = new_vel
                        vox_pc[picked_particles[i]] = new_pos

                        picked_velocity.append(velocity_his[picked_particles[i]])
                        picked_pos.append(new_pos)
                else:
                    picked_particles[i] = -1
        else:
            for i in range(self.args.datasenum_picker):
                new_picker_pos[i, :] = picker_pos[i, :] + action[i, :3]
                if pick_flag[i]:
                    if picked_particles[i] == -1:  # No particle is currently picked and thus need to select a particle to pick
                        dists = scipy.spatial.distance.cdist(picker_pos[i].reshape((-1, 3)), vox_pc[:, :3].reshape((-1, 3)))
                        picked_particles[i] = np.argmin(dists)
                    # update the position and velocity of the picked particle
                    if picked_particles[i] != -1:
                        old_pos = vox_pc[picked_particles[i]]
                        new_pos = vox_pc[picked_particles[i]] + new_picker_pos[i, :] - picker_pos[i, :]
                        new_vel = (new_pos - old_pos) / (self.dt * self.args.pred_time_interval)
                        tmp_vel_history = velocity_his[picked_particles[i]][3:].copy()
                        velocity_his[picked_particles[i], :-3] = tmp_vel_history
                        velocity_his[picked_particles[i], -3:] = new_vel

                        vox_pc[picked_particles[i]] = new_pos

                        picked_velocity.append(velocity_his[picked_particles[i]])
                        picked_pos.append(new_pos)
                else:
                    picked_particles[i] = int(-1)
        picked_status = (picked_velocity, picked_pos, new_picker_pos)

        return picked_particles, picked_status

    def _compute_node_attr(self, vox_pc, picked_points, velocity_his, **kwargs):
        # picked particle [0, 1]
        # normal particle [1, 0]
        node_one_hot = np.zeros((len(vox_pc), 2), dtype=np.float32)
        node_one_hot[:, 0] = 1
        for picked in picked_points:
            if picked != -1:
                node_one_hot[picked, 0] = 0
                node_one_hot[picked, 1] = 1

        if self.args.env_shape is not None:
            distance_to_shape, vector_to_shape = self._compute_distance_to_shape(vox_pc, kwargs['shape_pos'], kwargs['shape_size'], kwargs['shape_quat'])
        else:
            distance_to_shape = torch.from_numpy(vox_pc[:, 1]).view((-1, 1))
            vector_to_shape = torch.zeros([distance_to_shape.shape[0],3])
            vector_to_shape[:, 1] = 1

        node_one_hot = torch.from_numpy(node_one_hot)
        node_attr = torch.from_numpy(velocity_his)
        node_attr = torch.cat([node_attr, distance_to_shape, vector_to_shape, node_one_hot], dim=1)
        return node_attr
    def _compute_distance_to_shape(self):
        raise NotImplementedError

    def _compute_edge_attr(self, input_type, data):
        ##### add env specific graph components
        ## Edge attributes:
        # [1, 0] Distance based neighbor
        # [0, 1] Mesh edges
        # Calculate undirected edge list and corresponding relative edge attributes (distance vector + magnitude)
        vox_pc, velocity_his = data['pointcloud'], data['vel_his']

        point_tree = scipy.spatial.cKDTree(vox_pc)
        undirected_neighbors = np.array(list(point_tree.query_pairs(self.args.neighbor_radius, p=2))).T

        if len(undirected_neighbors) > 0:
            dist_vec = vox_pc[undirected_neighbors[0, :]] - vox_pc[undirected_neighbors[1, :]]
            dist = np.linalg.norm(dist_vec, axis=1, keepdims=True)
            edge_attr = np.concatenate([dist_vec, dist], axis=1)
            edge_attr_reverse = np.concatenate([-dist_vec, dist], axis=1)

            # Generate directed edge list and corresponding edge attributes
            edges = np.concatenate([undirected_neighbors, undirected_neighbors[::-1]], axis=1)
            edge_attr = np.concatenate([edge_attr, edge_attr_reverse])
            num_distance_edges = edges.shape[1]
        else:
            raise ValueError('undirected_neighbors is empty')

        # Build mesh edges -- both directions
        if self.args.use_mesh_edge:

            mesh_edges = data['mesh_edges']

            mesh_dist_vec = vox_pc[mesh_edges[0, :]] - vox_pc[mesh_edges[1, :]]
            mesh_dist = np.linalg.norm(mesh_dist_vec, axis=1, keepdims=True)
            mesh_edge_attr = np.concatenate([mesh_dist_vec, mesh_dist], axis=1)
            num_mesh_edges = mesh_edges.shape[1]

            # concatenate all edge attributes
            edge_attr = np.concatenate([edge_attr, mesh_edge_attr], axis=0) if num_distance_edges > 0 else mesh_edge_attr
            edge_attr, mesh_edges = torch.from_numpy(edge_attr), torch.from_numpy(mesh_edges)

            # Concatenate edge types
            edge_types = np.zeros((num_mesh_edges + num_distance_edges, 2), dtype=np.float32)
            edge_types[:num_distance_edges, 0] = 1.
            edge_types[num_distance_edges:, 1] = 1.
            edge_types = torch.from_numpy(edge_types)
            edge_attr = torch.cat([edge_attr, edge_types], dim=1)

            if num_distance_edges > 0:
                edges = torch.from_numpy(edges)
                edges = torch.cat([edges, mesh_edges], dim=1)
            else:
                edges = mesh_edges
        else:
            if num_distance_edges > 0:
                edges, edge_attr = torch.from_numpy(edges), torch.from_numpy(edge_attr)
            else:
                # manually add one edge for correct processing when there is no collision edges
                print("number of distance edges is 0! adding fake edges")
                edges = np.zeros((2, 2), dtype=np.uint8)
                edges[0][0] = 0
                edges[1][0] = 1
                edges[0][1] = 0
                edges[1][1] = 2
                edge_attr = np.zeros((2, self.args.relation_dim), dtype=np.float32)
                edges = torch.from_numpy(edges).bool()
                edge_attr = torch.from_numpy(edge_attr)
                print("shape of edges: ", edges.shape)
                print("shape of edge_attr: ", edge_attr.shape)
        return edges, edge_attr

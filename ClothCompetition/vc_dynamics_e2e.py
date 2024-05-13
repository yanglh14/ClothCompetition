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

import torch
import torch_geometric

from softgym.utils.visualization import save_numpy_as_gif

from ClothCompetition.models import GNN
from ClothCompetition.dataset_e2e import ClothDataset
from ClothCompetition.utils.data_utils import AggDict
from ClothCompetition.utils.utils import extract_numbers, pc_reward_model, visualize
from ClothCompetition.utils.camera_utils import get_matrix_world_to_camera, project_to_image
from ClothCompetition.utils.utils import downsample, load_data, load_data_list, store_h5_data, voxelize_pointcloud, \
    pc_reward_model
from ClothCompetition.utils.camera_utils import get_observable_particle_index, get_observable_particle_index_old, \
    get_world_coords, get_observable_particle_index_3
class VCDynamics(object):
    def __init__(self, args, env, vcd_edge=None):
        # Create Models
        self.args = args
        self.env = env
        self.train_mode = args.train_mode
        self.device = torch.device(self.args.cuda_idx)
        self.input_types = ['full', 'vsbl'] if self.train_mode == 'graph_imit' else [self.train_mode]
        self.models, self.optims, self.schedulers = {}, {}, {}
        for m in self.input_types:
            self.models[m] = GNN(args, decoder_output_dim=3, name=m, use_reward=False if self.train_mode == 'vsbl' else True)  # Predict acceleration
            lr = getattr(self.args, m + '_lr') if hasattr(self.args, m + '_lr') else self.args.lr
            self.optims[m] = torch.optim.Adam(self.models[m].param(), lr=lr, betas=(self.args.beta1, 0.999))
            self.schedulers[m] = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optims[m], 'min', factor=0.8,
                                                                            patience=3, verbose=True)
            self.models[m].to(self.device)

        self.vcd_edge = vcd_edge
        self.load_model(self.args.load_optim)

        print("VCD dynamics models created")
        if self.train_mode == 'graph_imit' and not args.tune_teach:
            self.models[self.input_types[0]].freeze()

        # Create Dataloaders
        self.datasets = {phase: ClothDataset(args, self.input_types, phase, env) for phase in ['train', 'valid']}
        for phase in ['train', 'valid']: self.datasets[phase].vcd_edge = self.vcd_edge

        follow_batch = ['x_{}'.format(t) for t in self.input_types]
        if not args.gen_data:

            self.dataloaders = {x: torch_geometric.data.DataLoader(
                self.datasets[x], batch_size=args.batch_size, follow_batch=follow_batch,
                shuffle=True if x == 'train' else False, drop_last=True,
                num_workers=args.num_workers, pin_memory=True, prefetch_factor=5 if args.num_workers > 0 else 2)
                for x in ['train', 'valid']}
        else:
            self.dataloaders = {x: None for x in ['train', 'valid']}

        self.mse_loss = torch.nn.MSELoss()
        self.log_dir = logger.get_dir()
        if self.args.use_wandb and args.eval == 0:
            # To use wandb, you need to create an account and run 'wandb login'.
            wandb.init(project='icra2024_cloth_competition', entity='yanglh14', name=args.exp_name, resume='allow',
                       id=None, settings=wandb.Settings(start_method='thread'))
            print('Weights & Biases is initialized with run name {}'.format(args.exp_name))
            wandb.config.update(args, allow_val_change=True)

    def retrieve_data(self, data, key):
        """ vsbl: [vsbl], full: [full], dual :[vsbl, full]  """
        identifier = '_{}'.format(key)
        out_data = {k.replace(identifier, ''): v for k, v in data.items() if identifier in k}
        return out_data

    def generate_dataset(self):
        os.system('mkdir -p ' + self.args.dataf)
        for phase in ['train', 'valid']:
            self.datasets[phase].generate_dataset()
        print('Dataset generated in', self.args.dataf)

    def resume_training(self):
        pass

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

    def train(self):
        # Training loop
        st_epoch = self.load_epoch if hasattr(self, 'load_epoch') else 0
        print('st epoch ', st_epoch)
        best_valid_loss = {m_name: np.inf for m_name in self.models}
        phases = ['train', 'valid'] if self.args.eval == 0 else ['valid']
        for epoch in range(st_epoch, self.args.n_epoch):
            for phase in phases:
                self.set_mode(phase)
                # Log all the useful metrics
                epoch_infos = {m: AggDict(is_detach=True) for m in self.models}

                epoch_len = len(self.dataloaders[phase])
                for i, data in tqdm(enumerate(self.dataloaders[phase]), desc=f'Epoch {epoch}, phase {phase}'):
                    data = data.to(self.device).to_dict()
                    iter_infos = {m_name: AggDict(is_detach=False) for m_name in self.models}
                    preds = {}
                    last_global = torch.zeros(self.args.batch_size, self.args.global_size, dtype=torch.float32,
                                              device=self.device)
                    with torch.set_grad_enabled(phase == 'train'):
                        for (m_name, model), iter_info in zip(self.models.items(), iter_infos.values()):
                            inputs = self.retrieve_data(data, m_name)
                            inputs['u'] = last_global
                            pred = model(inputs)
                            preds[m_name] = pred
                            # iter_info.add_item('accel_loss', self.mse_loss(pred['accel'], inputs['gt_accel']))
                            # iter_info.add_item('sqrt_accel_loss', torch.sqrt(iter_info['accel_loss']))
                            iter_info.add_item('reward_loss', self.mse_loss(pred['reward_end'], inputs['gt_reward_end']))
                            iter_info.add_item('sqrt_reward_loss', torch.sqrt(iter_info['reward_loss']))

                            if self.train_mode != 'vsbl':
                                iter_info.add_item('reward_loss',
                                                   self.mse_loss(pred['reward_nxt'].squeeze(), inputs['gt_reward_nxt']))


                    for m_name in self.models:
                        iter_info = iter_infos[m_name]
                        for feat in ['n_nxt', 'lat_nxt']:  # Node and global output
                            iter_info.add_item(feat + '_norm', torch.norm(preds[m_name][feat], dim=1).mean())

                        if self.args.train_mode == 'vsbl':  # Student loss
                            iter_info.add_item('total_loss', iter_info['reward_loss'])

                        if phase == 'train':
                            self.optims[m_name].zero_grad()
                            iter_info['total_loss'].backward()
                            self.optims[m_name].step()

                        epoch_infos[m_name].update_by_add(iter_infos[m_name])  # Aggregate info

                if phase == 'train' and epoch % self.args.save_model_interval == 0:
                    for m_name, model in self.models.items():
                        suffix = '{}'.format(epoch)
                        model.save_model(self.log_dir, m_name, suffix, self.optims[m_name])

                if phase == 'valid':
                    for m_name, model in self.models.items():
                        epoch_info = epoch_infos[m_name]
                        cur_loss = epoch_info[f"{m_name}/{phase}/" + 'total_loss']
                        if not self.args.fixed_lr:
                            self.schedulers[m_name].step(cur_loss)
                        if cur_loss < best_valid_loss[m_name]:
                            best_valid_loss[m_name] = cur_loss
                            state_dict = self.args.__dict__
                            state_dict['best_epoch'] = epoch
                            state_dict['best_valid_loss'] = cur_loss
                            with open(osp.join(self.log_dir, 'best_state.json'), 'w') as f:
                                json.dump(state_dict, f, indent=2, sort_keys=True)
                            model.save_model(self.log_dir, m_name, 'best', self.optims[m_name])
                # logging
                logger.record_tabular(phase + '/epoch', epoch)
                for m_name in self.models:
                    epoch_info = epoch_infos[m_name]
                    epoch_info = epoch_info.get_mean(f"{m_name}/{phase}/", epoch_len)
                    epoch_info['lr'] = self.optims[m_name].param_groups[0]['lr']
                    logger.log(
                        f'{phase} [{epoch}/{self.args.n_epoch}] Loss: {epoch_info[f"{m_name}/{phase}/total_loss"]:.4f}',
                        best_valid_loss[m_name])

                    for k, v in epoch_info.items():
                        logger.record_tabular(k, v)

                    if self.args.use_wandb and self.args.eval == 0:
                        wandb.log(epoch_info, step=epoch)

                logger.dump_tabular()

    def set_mode(self, mode='train'):
        for model in self.models.values():
            model.set_mode('train' if mode == 'train' else 'eval')

    def to(self, cuda_idx):
        for model in self.models.values():
            model.to(torch.device("cuda:{}".format(cuda_idx)))

    def rollout(self, args):
        """
        args need to contain the following contents:
            model_input_data: current point cloud, velocity history, picked point, picker position, etc
            actions: rollout actions
            reward_model: reward function
            cuda_idx (optional): default 0
            robot_exp (optional): default False

        return a dict:
            final_ret: final reward of the rollout
            model_positions: model predicted point cloud positions
            shape_positions: positions of the pickers, for visualization
            mesh_edges: predicted mesh edge
            time_cost: time cost for different parts of the rollout function
        """
        model_input_data = args['model_input_data']
        actions = args['actions']  # NOTE: sequence of actions to rollout
        reward_model = args['reward_model']
        m_name = args['m_name']
        dataset = self.datasets['train']  # Both train and valid are the same during inference
        H = len(actions)  # Planning horizon
        cuda_idx = args.get('cuda_idx', 0)
        robot_exp = args.get('robot_exp', False)

        self.set_mode('eval')
        self.to(cuda_idx)
        self.device = torch.device(cuda_idx)

        pc_pos = model_input_data['pointcloud']
        pc_vel_his = model_input_data['vel_his']
        picker_pos = model_input_data['picker_position']
        picked_points_idx = model_input_data['picked_points_idx'] # picked point index
        scene_params = model_input_data['scene_params']
        observable_particle_index = model_input_data['partial_pc_mapped_idx']
        rest_dist = model_input_data.get('rest_dist', None)
        mesh_edges = model_input_data.get('mesh_edges', None)
        assert rest_dist is None  # The rest_dist will be computed from the initial_particle_pos?

        # record model predicted point cloud positions
        model_positions = np.zeros((H, len(pc_pos), 3))
        shape_positions = np.zeros((H, 2, 3))
        initial_pc_pos = pc_pos.copy()
        pred_rewards = np.zeros(H)
        gt_pos_rewards = np.zeros(H)
        picked_points = []

        # Predict mesh during evaluation and use gt edges during training
        if self.vcd_edge is not None and mesh_edges is None:
            model_input_data['cuda_idx'] = cuda_idx
            mesh_edges = self.vcd_edge.infer_mesh_edges(model_input_data)

        # for ablation that uses first-time step collision edges as the mesh edges
        if self.datasets['train'].args.use_collision_as_mesh_edge:
            print("construct collision edges at the first time step as mesh edges!")
            neighbor_radius = self.datasets['train'].args.neighbor_radius
            point_tree = scipy.spatial.cKDTree(pc_pos)
            undirected_neighbors = np.array(list(point_tree.query_pairs(neighbor_radius, p=2))).T
            mesh_edges = np.concatenate([undirected_neighbors, undirected_neighbors[::-1]], axis=1)

        ret = 0
        for t in range(H):
            data = {'pointcloud': pc_pos,
                    'vel_his': pc_vel_his,
                    'picker_position': picker_pos,
                    'action': actions[t],
                    'picked_points_idx': picked_points_idx,
                    'scene_params': scene_params,
                    'partial_pc_mapped_idx': observable_particle_index if not robot_exp else range(len(pc_pos)),
                    'mesh_edges': mesh_edges,
                    'rest_dist': rest_dist,
                    'initial_particle_pos': initial_pc_pos}

            # for ablation that fixes the collision edge as those computed in the firs time step
            if not self.datasets['train'].args.fix_collision_edge:
                graph_data = dataset.build_graph(data, input_type=m_name, robot_exp=robot_exp)
            else:
                logger.log('using fixed collision edge!')
                if t == 0:  # store the initial collision edge
                    graph_data = dataset.build_graph(data, input_type=m_name, robot_exp=robot_exp)
                    fix_edges, fix_edge_attr = graph_data['neighbors'], graph_data['edge_attr']
                else:  # use the stored initial edge
                    graph_data = dataset.build_graph(data, input_type=m_name, robot_exp=robot_exp)
                    graph_data['neighbors'], graph_data['edge_attr'] = fix_edges, fix_edge_attr

            picked_points.append(graph_data['picked_particles'])
            model_positions[t] = pc_pos
            shape_positions[t] = picker_pos

            inputs = {'x': graph_data['node_attr'].to(self.device),
                      'edge_attr': graph_data['edge_attr'].to(self.device),
                      'edge_index': graph_data['neighbors'].to(self.device),
                      'x_batch': torch.zeros(graph_data['node_attr'].size(0), dtype=torch.long, device=self.device),
                      'u': torch.zeros([1, self.args.global_size], device=self.device)}

            # obtain model predictions
            with torch.no_grad():
                pred = self.models[m_name](inputs)
                pred_accel = pred['accel'].cpu().numpy()
                pred_reward = pred['reward_nxt'].cpu().numpy() if 'reward_nxt' in pred else 0.

            pc_pos, pc_vel_his, picker_pos = self.update_graph(pred_accel, pc_pos, pc_vel_his,
                                                               graph_data['picked_status'],
                                                               graph_data['picked_particles'])
            reward = reward_model(pc_pos)
            ret += reward

            pred_rewards[t] = pred_reward
            gt_pos_rewards[t] = reward

            if t == H - 1:
                final_ret = reward
        if mesh_edges is None:  # No mesh edges input during training
            mesh_edges = data['mesh_edges']  # This is modified inside prepare_transition function
        return dict(final_ret=final_ret,
                    model_positions=model_positions,
                    shape_positions=shape_positions,
                    mesh_edges=mesh_edges,
                    pred_rewards=pred_rewards,
                    gt_pos_rewards=gt_pos_rewards,
                    picked_points = picked_points)

    def update_graph(self, pred_accel, pc_pos, velocity_his, picked_status, picked_particles):
        """ Euler integration"""
        # vel_his: [v(t-20), ... v(t)], v(t) = (p(t) - o(t-5)) / (5*dt)
        pred_time_interval = self.args.pred_time_interval
        pred_vel = velocity_his[:, -3:] + pred_accel * self.args.dt * pred_time_interval
        pc_pos = pc_pos + pred_vel * self.args.dt * pred_time_interval

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

    def infer_performance(self):
        data_dir = './data/base_v5/train/'
        # load data
        idx_rollout = 0
        idx_timestep = 0
        idx_end = 1
        data_names = ['positions',  # Position and velocity of each simulation particle, N x 3 float
                      'picker_position',  # Position of all pickers
                      'scene_params',  # [cloth_particle_radius, xdim, ydim, config_id]
                      'downsample_idx',  # Indexes of the down-sampled particles
                      'downsample_observable_idx',
                      'observable_idx',  # Indexes of the observed particles
                      'pointcloud',
                      'picked_particles']  # point cloud position by back-projecting the depth image
        data_cur = load_data(data_dir, idx_rollout, idx_timestep, data_names)
        data_end = load_data(data_dir, idx_rollout, idx_end, data_names)

        pointcloud = data_cur['pointcloud']
        vox_pc = voxelize_pointcloud(pointcloud, self.args.voxel_size)

        partial_particle_pos = data_cur['positions'][data_cur['downsample_idx']][
            data_cur['downsample_observable_idx']]

        # find the grapsed points
        picked_paticles = data_cur['picked_particles']
        picked_positions = data_cur['positions'][picked_paticles]
        vox_pc = np.concatenate([vox_pc, picked_positions], axis=0)
        picked_points_idx = np.array([len(vox_pc) - 2, len(vox_pc) - 1])

        # Use clean observable point cloud for bi-partite matching
        # particle_pc_mapped_idx: For each point in pc, give the index of the closest point on the visible downsample mesh
        vox_pc, partial_pc_mapped_idx = get_observable_particle_index_3(vox_pc, partial_particle_pos,
                                                                        threshold=self.args.voxel_size)

        partial_pc_mapped_idx = data_cur['downsample_observable_idx'][
            partial_pc_mapped_idx]  # Map index from the observable downsampled mesh to the downsampled mesh

        downsample_idx = data_cur['downsample_idx']
        full_pos_cur = data_cur['positions']

        gt_reward_crt = torch.FloatTensor([pc_reward_model(full_pos_cur[downsample_idx])])
        gt_reward_end = torch.FloatTensor([pc_reward_model(data_end['positions'][downsample_idx])])
        normalized_vox_pc = vox_pc - np.mean(vox_pc, axis=0)

        data = {'pointcloud': normalized_vox_pc,
                'vel_his': np.zeros((len(vox_pc), 15)),
                'picker_position': np.zeros([2,3]),
                'action': np.array([0,0,0,1,0,0,0,1]),
                'picked_points_idx': picked_points_idx,
                'partial_pc_mapped_idx': np.arange(len(vox_pc)),
                }


        model_input_data = dict(
            pointcloud=vox_pc,
            cuda_idx=0,
        )
        mesh_edges = self.vcd_edge.infer_mesh_edges(model_input_data)
        data['mesh_edges'] = mesh_edges


        graph_data = self.datasets['valid'].build_graph(data, input_type='vsbl', robot_exp=False)

        inputs = {'x': graph_data['node_attr'].to(self.device),
                  'edge_attr': graph_data['edge_attr'].to(self.device),
                  'edge_index': graph_data['neighbors'].to(self.device),
                  'x_batch': torch.zeros(graph_data['node_attr'].size(0), dtype=torch.long, device=self.device),
                  'u': torch.zeros([1, self.args.global_size], device=self.device)}

        # obtain model predictions
        with torch.no_grad():
            pred = self.models['vsbl'](inputs)
            pred_reward = pred['reward_end'].cpu().numpy()
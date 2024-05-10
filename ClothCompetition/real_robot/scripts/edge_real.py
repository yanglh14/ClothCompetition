import numpy as np
import os

from ADFM.module.models import GNN

from scipy import spatial
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import wandb
from omegaconf import OmegaConf

import os.path as osp

from chester import logger
from ADFM.utils.camera_utils import get_matrix_world_to_camera, project_to_image
import matplotlib.pyplot as plt
import torch_geometric

from ADFM.module.dataset_edge import ClothDatasetPointCloudEdge
from ADFM.utils.utils import extract_numbers
from ADFM.utils.data_utils import AggDict
import json
from tqdm import tqdm

class EdgeReal(object):
    def __init__(self, args, env=None):
        self.args = args
        self.env = env
        self.model = GNN(args.model, decoder_output_dim=1, name='EdgeGNN')  # Predict 0/1 Label for mesh edge classification
        self.device = torch.device(self.args.cuda_idx)
        self.model.to(self.device)
        self.optim = torch.optim.Adam(self.model.param(), lr=args.lr, betas=(args.beta1, 0.999))
        self.scheduler = ReduceLROnPlateau(self.optim, 'min', factor=0.8, patience=3, verbose=True)
        if self.args.edge_model_path is not None:
            self.load_model(self.args.load_optim)

        self.log_dir = logger.get_dir()
        self.bce_logit_loss = nn.BCEWithLogitsLoss()
        self.load_epoch = 0

    def infer_mesh_edges(self, args):
        """
        args: a dict
            scene_params
            pointcloud
            cuda_idx
        """
        point_cloud = args['pointcloud']
        cuda_idx = args.get('cuda_idx', 0)

        self.set_mode('eval')
        if cuda_idx >= 0:
            self.to(cuda_idx)

        normalized_point_cloud = point_cloud - np.mean(point_cloud, axis=0)
        data_ori = {
            'normalized_vox_pc': normalized_point_cloud,
        }
        data = self.build_graph(data_ori, get_gt_edge_label=False)
        with torch.no_grad():
            data['x_batch'] = torch.zeros(data['x'].size(0), dtype=torch.long, device=self.device)
            data['u'] = torch.zeros([1, self.args.model.global_size], device=self.device)
            for key in ['x', 'edge_index', 'edge_attr']:
                data[key] = data[key].to(self.device)
            pred_mesh_edge_logits = self.model(data)['mesh_edge']

        pred_mesh_edge_logits = pred_mesh_edge_logits.cpu().numpy()
        pred_mesh_edge = pred_mesh_edge_logits > 0

        edges = data['edge_index'].detach().cpu().numpy()
        senders = []
        receivers = []
        num_edges = edges.shape[1]
        for e_idx in range(num_edges):
            if pred_mesh_edge[e_idx]:
                senders.append(int(edges[0][e_idx]))
                receivers.append(int(edges[1][e_idx]))

        mesh_edges = np.vstack([senders, receivers])
        return mesh_edges

    def build_graph(self, data, get_gt_edge_label=True):
        """
        data: positions, picked_points, picked_point_positions, scene_params
        downsample: whether to downsample the graph
        test: if False, we are in the training mode, where we know exactly the picked point and its movement
            if True, we are in the test mode, we have to infer the picked point in the (downsampled graph) and compute
                its movement.

        return:
        node_attr: N x (vel_history x 3)
        edges: 2 x E, the edges
        edge_attr: E x edge_feature_dim
        gt_mesh_edge: 0/1 label for groundtruth mesh edge connection.
        """
        node_attr = torch.from_numpy(data['normalized_vox_pc'])
        edges, edge_attr = self._compute_edge_attr(data['normalized_vox_pc'])

        if get_gt_edge_label:
            raise NotImplementedError
        else:
            gt_mesh_edge = None

        return {
            'x': node_attr,
            'edge_index': edges,
            'edge_attr': edge_attr,
            'gt_mesh_edge': gt_mesh_edge
        }

    def _compute_edge_attr(self, vox_pc):
        point_tree = spatial.cKDTree(vox_pc)
        undirected_neighbors = np.array(list(point_tree.query_pairs(self.args.dataset.neighbor_radius, p=2))).T

        if len(undirected_neighbors) > 0:
            dist_vec = vox_pc[undirected_neighbors[0, :]] - vox_pc[undirected_neighbors[1, :]]
            dist = np.linalg.norm(dist_vec, axis=1, keepdims=True)
            edge_attr = np.concatenate([dist_vec, dist], axis=1)
            edge_attr_reverse = np.concatenate([-dist_vec, dist], axis=1)

            # Generate directed edge list and corresponding edge attributes
            edges = torch.from_numpy(np.concatenate([undirected_neighbors, undirected_neighbors[::-1]], axis=1))
            edge_attr = torch.from_numpy(np.concatenate([edge_attr, edge_attr_reverse]))
        else:
            print("number of distance edges is 0! adding fake edges")
            edges = np.zeros((2, 2), dtype=np.uint8)
            edges[0][0] = 0
            edges[1][0] = 1
            edges[0][1] = 0
            edges[1][1] = 2
            edge_attr = np.zeros((2, self.args.dataset.relation_dim), dtype=np.float32)
            edges = torch.from_numpy(edges).bool()
            edge_attr = torch.from_numpy(edge_attr)
            print("shape of edges: ", edges.shape)
            print("shape of edge_attr: ", edge_attr.shape)

        return edges, edge_attr

    def load_model(self, load_optim=False):
        self.model.load_model(self.args.edge_model_path, load_optim=load_optim, optim=self.optim)
        self.load_epoch = extract_numbers(self.args.edge_model_path)[-1]

    def to(self, cuda_idx):
        self.model.to(torch.device("cuda:{}".format(cuda_idx)))

    def set_mode(self, mode='train'):
        self.model.set_mode('train' if mode == 'train' else 'eval')
from .finalmlp import FinalPredMLP
from .selfatten import SelfAttentionLayer
from .subgraph import SubGraph
import os
import pdb
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from torch_geometric.nn import MessagePassing, max_pool
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch
from torch_geometric.data import Data, DataLoader
import sys
sys.path.append('..')
# from core.utils import latent_code_dim


def latent_code_dim(latent_code):
    """
    case of latent code
    latent_code:
        winding_angle:
            type: categorical
            dim: 2
        speed:
            type: continuous
            range: [0.0, 1.0]
    """
    dim = 0
    for _, value in latent_code.items():
        if value['type'] == 'categorical':
            dim += value['dim']
        elif value['type'] == 'continuous':
            dim += 1
    assert dim > 0, "latent_code must have at least one dimension"
    return dim


class HGNN_Info(nn.Module):
    """
    hierarchical GNN with trajectory prediction MLP
    """

    def __init__(self, in_channels, latent_code, out_channels, goal_shape=2, num_subgraph_layers=2, num_global_graph_layer=1, subgraph_width=32, global_graph_width=32, final_mlp_hidden_width=64):
        super(HGNN_Info, self).__init__()
        self.goal_shape = goal_shape
        self.latent_code = latent_code
        self.polyline_vec_shape = in_channels * (2 ** num_subgraph_layers)

        self.subgraph = SubGraph(in_channels, num_subgraph_layers, subgraph_width)
        self.self_atten_layer = SelfAttentionLayer(self.polyline_vec_shape, global_graph_width, need_scale=False)

        # specific for combining latent code
        self.dim_c = latent_code_dim(self.latent_code)
        self.traj_pred_mlp = FinalPredMLP(global_graph_width + self.goal_shape + self.dim_c, out_channels, final_mlp_hidden_width)

    def forward(self, data, c):
        """
        args: 
            data (Data): [x, y, cluster, edge_index, valid_len,goal,last_speed]
            c (tensor): [dim_c]
        """
        time_step_len = int(data.time_step_len[0])
        valid_lens = data.valid_len
        sub_graph_out = self.subgraph(data)
        x = sub_graph_out.x.view(-1, time_step_len, self.polyline_vec_shape)
        out = self.self_atten_layer(x, valid_lens)
        out_new=torch.cat((out[:, [0]].squeeze(1),data.goal.view(-1,2),c.view(-1,self.dim_c)),dim=1)
        pred = self.traj_pred_mlp(out_new)
        return pred
    

# function as discriminator and Q function
class HGNN_Disc_Info(nn.Module):
    """
    hierarchical GNN with trajectory prediction MLP
    """

    def __init__(self, in_channels, action_dim, latent_code, out_channels=1, goal_shape=2, num_subgraph_layers=2, num_global_graph_layer=1, subgraph_width=32, global_graph_width=32, final_mlp_hidden_width=64):
        super(HGNN_Disc_Info, self).__init__()
        self.goal_shape = goal_shape
        self.action_dim = action_dim
        self.latent_code = latent_code
        self.polyline_vec_shape = in_channels * (2 ** num_subgraph_layers)


        self.subgraph = SubGraph(in_channels, num_subgraph_layers, subgraph_width)
        self.self_atten_layer = SelfAttentionLayer(self.polyline_vec_shape, global_graph_width, need_scale=False)
        # last part for discriminator
        self.traj_pred_mlp = FinalPredMLP(global_graph_width + self.goal_shape + self.action_dim, out_channels, final_mlp_hidden_width)

    def forward(self, state, action):
        """
        args: 
            state (Data): [x, cluster, edge_index, valid_len,goal,last_speed]
            action
        """
        time_step_len = int(state.time_step_len[0])
        valid_lens = state.valid_len
        sub_graph_out = self.subgraph(state)
        x = sub_graph_out.x.view(-1, time_step_len, self.polyline_vec_shape)
        out = self.self_atten_layer(x, valid_lens)
        out_new=torch.cat((out[:, [0]].squeeze(1), state.goal.view(-1,2), action.view(-1,2)), dim=1)
        # output for discriminator
        pred = self.traj_pred_mlp(out_new)

        return pred


    


# function as Q function
class HGNN_Q_Info(nn.Module):
    """
    hierarchical GNN with trajectory prediction MLP
    """

    def __init__(self, in_channels, action_dim, dim_c, q_len_his, out_channels=1, goal_shape=2, num_subgraph_layers=2, num_global_graph_layer=1, subgraph_width=32, global_graph_width=32, final_mlp_hidden_width=64):
        super(HGNN_Q_Info, self).__init__()

        self.goal_shape = goal_shape
        self.action_dim = action_dim
        self.dim_c= dim_c
        self.len_his = q_len_his

        self.polyline_vec_shape = in_channels * (2 ** num_subgraph_layers)
        self.subgraph = SubGraph(in_channels, num_subgraph_layers, subgraph_width)
        self.self_atten_layer = SelfAttentionLayer(self.polyline_vec_shape, global_graph_width, need_scale=False)
        # last part for Q function
        self.fianal_classifier = FinalPredMLP(final_mlp_hidden_width, dim_c, final_mlp_hidden_width)
        self.conti_classifier = nn.GRU(global_graph_width+self.goal_shape+self.action_dim, final_mlp_hidden_width, batch_first=True)

    def forward(self, state, action, dones):
        """
        args: 
            state (Data): [x, cluster, edge_index, valid_len,goal,last_speed]
            action
        """
        time_step_len = int(state.time_step_len[0])
        valid_lens = state.valid_len
        sub_graph_out = self.subgraph(state)
        x = sub_graph_out.x.view(-1, time_step_len, self.polyline_vec_shape)
        out = self.self_atten_layer(x, valid_lens)
        out_new=torch.cat((out[:, [0]].squeeze(1), state.goal.view(-1,2), action.view(-1,2)), dim=1)

        if dones is not None:
            c = self.his_calculate_c(out_new, dones)
            return c


    def his_calculate_c(self,embeddings,dones):

        # seperate the input of GRU by dones
        # and calculate the c for each segment
        sep_indexs = torch.nonzero(dones.squeeze(1)).squeeze(1)
        start_index = 0
        cs = torch.empty((0,self.dim_c)).to(embeddings.device)
        for end_index in sep_indexs:
            cur_ptr = start_index
            while(cur_ptr <= end_index):
                star_ptr = max(0,cur_ptr-self.len_his+1,start_index)
                seq_embeddings = embeddings[star_ptr:cur_ptr+1,:]
                _, hn = self.conti_classifier(seq_embeddings)
                c = self.fianal_classifier(hn)
                cs = torch.cat((cs,c),dim=0)
                cur_ptr += 1
            start_index = end_index + 1
        # if exist the last segment
        if start_index < embeddings.shape[0]:
            cur_ptr = start_index
            while(cur_ptr < embeddings.shape[0]):
                star_ptr = max(0,cur_ptr-self.len_his+1,start_index)
                seq_embeddings = embeddings[star_ptr:cur_ptr+1,:]
                _, hn = self.conti_classifier(seq_embeddings)
                c = self.fianal_classifier(hn)
                cs = torch.cat((cs,c),dim=0)
                cur_ptr += 1
        return cs

        

















# function as Q function encoder
class HGNN_Q_ENCODER(nn.Module):
    """
    hierarchical GNN with trajectory prediction MLP
    """

    def __init__(self, in_channels, action_dim, goal_shape=2, num_subgraph_layers=2, num_global_graph_layer=1, subgraph_width=32, global_graph_width=32, final_mlp_hidden_width=64):
        super(HGNN_Q_ENCODER, self).__init__()
        self.goal_shape = goal_shape
        self.action_dim = action_dim
        self.polyline_vec_shape = in_channels * (2 ** num_subgraph_layers)


        self.subgraph = SubGraph(in_channels, num_subgraph_layers, subgraph_width)
        self.self_atten_layer = SelfAttentionLayer(self.polyline_vec_shape, global_graph_width, need_scale=False)
        
    def forward(self, state, action):
        """
        args: 
            state (Data): [x, cluster, edge_index, valid_len,goal,last_speed]
            action
        """
        time_step_len = int(state.time_step_len[0])
        valid_lens = state.valid_len
        sub_graph_out = self.subgraph(state)
        x = sub_graph_out.x.view(-1, time_step_len, self.polyline_vec_shape)
        out = self.self_atten_layer(x, valid_lens)
        out_new=torch.cat((out[:, [0]].squeeze(1), state.goal.view(-1,2), action.view(-1,2)), dim=1)

        return out_new





class HGNN_Info_ENCODER(nn.Module):
    """
    hierarchical GNN with trajectory prediction MLP
    """

    def __init__(self, in_channels, out_channels, latent_code, goal_shape=2, num_subgraph_layers=2, num_global_graph_layer=1, subgraph_width=32, global_graph_width=32, final_mlp_hidden_width=64):
        super(HGNN_Info_ENCODER, self).__init__()
        self.goal_shape = goal_shape

        self.latent_code = latent_code
        self.dim_c = latent_code_dim(self.latent_code)
        
        self.polyline_vec_shape = in_channels * (2 ** num_subgraph_layers)

        self.subgraph = SubGraph(in_channels, num_subgraph_layers, subgraph_width)
        self.self_atten_layer = SelfAttentionLayer(self.polyline_vec_shape, global_graph_width, need_scale=False)


    def forward(self, data, c):
        """
        args: 
            data (Data): [x, y, cluster, edge_index, valid_len,goal,last_speed]
            c (tensor): [dim_c]
        """
        time_step_len = int(data.time_step_len[0])
        valid_lens = data.valid_len
        sub_graph_out = self.subgraph(data)
        x = sub_graph_out.x.view(-1, time_step_len, self.polyline_vec_shape)
        out = self.self_atten_layer(x, valid_lens)
        out_new=torch.cat((out[:, [0]].squeeze(1),data.goal.view(-1,2),c.view(-1,self.dim_c)),dim=1)

        return out_new
    
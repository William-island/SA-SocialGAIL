import torch
from torch import nn
import math

from .utils import build_mlp, reparameterize, evaluate_lop_pi
from .GNN_modules import HGNN, HGNN_Info, HGNN_Info_ENCODER
from .GNN_modules.vectornet_info import latent_code_dim
from .GNN_modules.finalmlp import FinalPredMLP


class StateIndependentPolicy(nn.Module):

    def __init__(self, state_shape, action_shape, hidden_units=(64, 64),
                 hidden_activation=nn.Tanh()):
        super().__init__()

        self.net = build_mlp(
            input_dim=state_shape[0],
            output_dim=action_shape[0],
            hidden_units=hidden_units,
            hidden_activation=hidden_activation
        )
        self.log_stds = nn.Parameter(torch.zeros(1, action_shape[0]))

    def forward(self, states):
        return torch.tanh(self.net(states))

    def sample(self, states):
        return reparameterize(self.net(states), self.log_stds)

    def evaluate_log_pi(self, states, actions):
        return evaluate_lop_pi(self.net(states), self.log_stds, actions)









class GraphStateIndependentPolicy(nn.Module):

    def __init__(self, in_channels, action_shape, final_mlp_hidden_width=64):
        super().__init__()

        self.net = HGNN(in_channels, action_shape[0], final_mlp_hidden_width=final_mlp_hidden_width)
        self.log_stds = nn.Parameter(torch.zeros(1, action_shape[0]))

    def forward(self, states):
        return torch.tanh(self.net(states))

    def sample(self, states):
        return reparameterize(self.net(states), self.log_stds)

    def evaluate_log_pi(self, states, actions):
        return evaluate_lop_pi(self.net(states), self.log_stds, actions)





class GraphStateIndependentPolicy_Info(nn.Module):

    def __init__(self, in_channels, latent_code, action_shape, final_mlp_hidden_width=64):
        super().__init__()

        self.net = HGNN_Info(in_channels, latent_code, action_shape[0], final_mlp_hidden_width=final_mlp_hidden_width)
        self.log_stds = nn.Parameter(torch.zeros(1, action_shape[0]))

    def forward(self, states, cs):
        return torch.tanh(self.net(states,cs))

    def sample(self, states, cs):
        return reparameterize(self.net(states,cs), self.log_stds)

    def evaluate_log_pi(self, states, cs, actions):
        return evaluate_lop_pi(self.net(states,cs), self.log_stds, actions)



# DA version
class GraphStateIndependentPolicy_Info_DA(nn.Module):

    def __init__(self, in_channels, latent_code, action_shape, goal_shape=2, global_graph_width=32, final_mlp_hidden_width=64):
        super().__init__()

        code_dim = latent_code_dim(latent_code)

        self.net = HGNN_Info_ENCODER(in_channels, action_shape[0], latent_code, global_graph_width=global_graph_width, final_mlp_hidden_width=final_mlp_hidden_width)
        self.pred_mlp = FinalPredMLP(global_graph_width + goal_shape + code_dim, action_shape[0], final_mlp_hidden_width)
        
        self.log_stds = nn.Parameter(torch.zeros(1, action_shape[0]))

        # init the pred_mlp first layer
        # latent_code related weight to zero
        with torch.no_grad():
            self.pred_mlp.mlp[0].weight[:, -code_dim:].zero_()
            print("init actor's first layer latent_code related weight to zero!")

        

    def forward(self, states, cs):
        x = self.net(states,cs)
        return torch.tanh(self.pred_mlp(x))

    def sample(self, states, cs):
        return reparameterize(self(states,cs), self.log_stds)

    def evaluate_log_pi(self, states, cs, actions):
        return evaluate_lop_pi(self(states,cs), self.log_stds, actions)



class GraphStateIndependentPolicy_Info_DA_FixedStd(nn.Module):

    def __init__(self, in_channels, latent_code, action_shape, goal_shape=2, global_graph_width=32, final_mlp_hidden_width=64):
        super().__init__()

        code_dim = latent_code_dim(latent_code)

        self.net = HGNN_Info_ENCODER(in_channels, action_shape[0], latent_code, global_graph_width=global_graph_width, final_mlp_hidden_width=final_mlp_hidden_width)
        self.pred_mlp = FinalPredMLP(global_graph_width + goal_shape + code_dim, action_shape[0], final_mlp_hidden_width)
        
        # --- 修改部分开始 ---
        # 计算对应方差 0.03 的 log_std
        target_variance = 0.03
        fixed_log_std = 0.5 * math.log(target_variance)
        
        # 使用 register_buffer 而不是 nn.Parameter
        # 理由：buffer 不会被优化器更新，但会随模型一起移动到 GPU，并随 model.save() 保存
        self.register_buffer('log_stds', torch.full((1, action_shape[0]), fixed_log_std))
        # --- 修改部分结束 ---

        # init the pred_mlp first layer
        with torch.no_grad():
            self.pred_mlp.mlp[0].weight[:, -code_dim:].zero_()
            print("init actor's first layer latent_code related weight to zero!")

    def forward(self, states, cs):
        x = self.net(states, cs)
        # 这里输出的是均值 mu
        return torch.tanh(self.pred_mlp(x))

    def sample(self, states, cs):
        # 使用固定的 self.log_stds 进行采样
        return reparameterize(self(states, cs), self.log_stds)

    def evaluate_log_pi(self, states, cs, actions):
        # 使用固定的 self.log_stds 计算似然
        return evaluate_lop_pi(self(states, cs), self.log_stds, actions)















class StateDependentPolicy(nn.Module):

    def __init__(self, state_shape, action_shape, hidden_units=(256, 256),
                 hidden_activation=nn.ReLU(inplace=True)):
        super().__init__()

        self.net = build_mlp(
            input_dim=state_shape[0],
            output_dim=2 * action_shape[0],
            hidden_units=hidden_units,
            hidden_activation=hidden_activation
        )

    def forward(self, states):
        return torch.tanh(self.net(states).chunk(2, dim=-1)[0])

    def sample(self, states):
        means, log_stds = self.net(states).chunk(2, dim=-1)
        return reparameterize(means, log_stds.clamp_(-20, 2))

import torch
from torch import nn

from .utils import build_mlp
from .GNN_modules import HGNN, HGNN_Info, HGNN_Info_ENCODER
from .GNN_modules.vectornet_info import latent_code_dim
from .GNN_modules.finalmlp import FinalPredMLP


class StateFunction(nn.Module):

    def __init__(self, state_shape, hidden_units=(64, 64),
                 hidden_activation=nn.Tanh()):
        super().__init__()

        self.net = build_mlp(
            input_dim=state_shape[0],
            output_dim=1,
            hidden_units=hidden_units,
            hidden_activation=hidden_activation
        )

    def forward(self, states):
        return self.net(states)
    

class GraphStateFunction_Info(nn.Module):

    def __init__(self, in_channels, latent_code, final_mlp_hidden_width=64):
        super().__init__()

        self.net = HGNN_Info(in_channels, latent_code, 1, final_mlp_hidden_width=final_mlp_hidden_width)

    def forward(self, states, cs):
        return self.net(states,cs)

# special for diversity-aware rl value estimate
class GraphStateFunction_Info_DA(nn.Module):

    def __init__(self, in_channels, latent_code, goal_shape=2, global_graph_width=32, final_mlp_hidden_width=64):
        super().__init__()

        self.net = HGNN_Info_ENCODER(in_channels, 1, latent_code, global_graph_width=global_graph_width, final_mlp_hidden_width=final_mlp_hidden_width)

        code_dim = latent_code_dim(latent_code)

        ## two head: one for main reward value estimate, one for auxiliary reward (diversity reward) value estimate
        self.head_main = FinalPredMLP(global_graph_width + goal_shape + code_dim, 1, final_mlp_hidden_width)
        self.head_aux = FinalPredMLP(global_graph_width + goal_shape + code_dim, 1, final_mlp_hidden_width)

        # latent_code related weight to zero
        with torch.no_grad():
            self.head_main.mlp[0].weight[:, -code_dim:].zero_()
            print("init critic head_main's first layer latent_code related weight to zero!")

    def forward(self, states, cs):
        x = self.net(states,cs)
        return self.head_main(x), self.head_aux(x)


class GraphStateFunction(nn.Module):

    def __init__(self, in_channels, final_mlp_hidden_width=64):
        super().__init__()

        self.net = HGNN(in_channels, 1, final_mlp_hidden_width=final_mlp_hidden_width)

    def forward(self, states):
        return self.net(states)
    



class StateActionFunction(nn.Module):

    def __init__(self, state_shape, action_shape, hidden_units=(100, 100),
                 hidden_activation=nn.Tanh()):
        super().__init__()

        self.net = build_mlp(
            input_dim=state_shape[0] + action_shape[0],
            output_dim=1,
            hidden_units=hidden_units,
            hidden_activation=hidden_activation
        )

    def forward(self, states, actions):
        return self.net(torch.cat([states, actions], dim=-1))


class TwinnedStateActionFunction(nn.Module):

    def __init__(self, state_shape, action_shape, hidden_units=(256, 256),
                 hidden_activation=nn.ReLU(inplace=True)):
        super().__init__()

        self.net1 = build_mlp(
            input_dim=state_shape[0] + action_shape[0],
            output_dim=1,
            hidden_units=hidden_units,
            hidden_activation=hidden_activation
        )
        self.net2 = build_mlp(
            input_dim=state_shape[0] + action_shape[0],
            output_dim=1,
            hidden_units=hidden_units,
            hidden_activation=hidden_activation
        )

    def forward(self, states, actions):
        xs = torch.cat([states, actions], dim=-1)
        return self.net1(xs), self.net2(xs)

    def q1(self, states, actions):
        return self.net1(torch.cat([states, actions], dim=-1))

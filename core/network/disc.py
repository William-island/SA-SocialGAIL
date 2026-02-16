import torch
from torch import nn
import torch.nn.functional as F

from .utils import build_mlp
from .GNN_modules import HGNN_Disc, HGNN_Disc_Info, HGNN_Q_ENCODER #, HGNN_Q
from .GNN_modules.finalmlp import FinalPredMLP
import math


class GAILDiscrim(nn.Module):

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
    
    def gp_forward(self,data):
        return self.net(data)

    def calculate_reward(self, states, actions):
        # PPO(GAIL) is to maximize E_{\pi} [-log(1 - D)].
        with torch.no_grad():
            # return -F.logsigmoid(-self.forward(states, actions))
            return self.forward(states, actions)
        








class GraphDiscrim_Info(nn.Module):
    def __init__(self, in_channels, action_shape, latent_code, final_mlp_hidden_width=64):
        super(GraphDiscrim_Info, self).__init__()
        self.graph_model = HGNN_Disc_Info(in_channels, action_shape[0], latent_code, final_mlp_hidden_width=final_mlp_hidden_width)

        self.reward_i_coef = 1.0
        # self.reward_us_coef = 0.1 # 0.1 # 1.0

    def forward(self, state, action):
            x = self.graph_model(state, action)
            return x
    
    def calculate_reward(self, states, actions):
        # PPO(GAIL) is to maximize E_{\pi} [-log(1 - D)].
        with torch.no_grad():
            ## origin reward
            # return -F.logsigmoid(-self.forward(states, actions))

            r = self.forward(states, actions)
            ## reward
            rewards_i = self.reward_i_coef * r
            return rewards_i

class GraphDiscrim(nn.Module):
    def __init__(self, in_channels, action_shape, final_mlp_hidden_width=64):
        super(GraphDiscrim, self).__init__()
        self.graph_model = HGNN_Disc(in_channels, action_shape[0], final_mlp_hidden_width=final_mlp_hidden_width)

    def forward(self, state, action):
        x = self.graph_model(state, action)
        return x
    
    def calculate_reward(self, states, actions):
        # PPO(GAIL) is to maximize E_{\pi} [-log(1 - D)].
        with torch.no_grad():
            # return -F.logsigmoid(-self.forward(states, actions))
            return self.forward(states, actions)






















## New version of Q function
# use HGNN_Q_ENCODER
# with one head for Q value,
# one head for feature predition
class SGI_GraphQ(nn.Module):
    def __init__(self, in_channels, action_shape, latent_code, goal_shape=2, final_mlp_hidden_width=64, global_graph_width=32):
        super(SGI_GraphQ, self).__init__()

        # encode state and action
        self.graph_model = HGNN_Q_ENCODER(in_channels, action_shape[0], final_mlp_hidden_width=final_mlp_hidden_width)

        # Q value head after encoder
        if latent_code['type'] == 'categorical':
            code_dim = latent_code['dim']
        elif latent_code['type'] == 'continuous':
            code_dim = 2
        self.latent_code = latent_code
        
        ## one choice
        # self.q_head = FinalPredMLP(global_graph_width + goal_shape + action_shape[0], code_dim, final_mlp_hidden_width)
        # self.feature_head = nn.Linear(global_graph_width + goal_shape + action_shape[0], latent_code['feature_dim'])

        ## one choice
        self.q_head = nn.Linear(global_graph_width + goal_shape + action_shape[0], code_dim)
        self.feature_head = nn.Linear(global_graph_width + goal_shape + action_shape[0], latent_code['feature_dim'])


        # # only one layer
        # self.q_head = nn.Linear(global_graph_width + goal_shape + action_shape[0], code_dim)

        # # feature prediction head after encoder
        # self.feature_head = FinalPredMLP(global_graph_width + goal_shape + action_shape[0], latent_code['feature_dim'], final_mlp_hidden_width)


    def forward(self, state, action):
        x = self.graph_model(state, action)
        return x
    


    
    def predict_feature(self, state, action):
        x = self.forward(state, action)
        return self.feature_head(x)

    def calculate_feature_loss(self, states, actions, target_features):
        # Predict features from encoder and compare to handcrafted targets
        pred_feat = self.predict_feature(states, actions)
        loss_f = F.mse_loss(pred_feat, target_features)
        return loss_f



    def predict_q(self, state, action):
        x = self.forward(state, action)
        q_values = self.q_head(x)
        return q_values

    def calculate_loss(self, states, actions, code_values):
        q_values = self.predict_q(states, actions)
        
        # calculate loss
        if self.latent_code['type'] == 'categorical':
            loss = F.cross_entropy(q_values, torch.argmax(code_values, dim=1))
        elif self.latent_code['type'] == 'continuous':
            mu, log_var = q_values.chunk(2, dim=-1)
            log_var = torch.clamp(log_var, min=math.log(0.05**2), max=math.log(1.0**2))
            var = log_var.exp()
            # 负对数似然
            loss = 0.5 * ((code_values - mu) ** 2 / var + log_var).mean()

        return loss

    

    def calculate_reward(self, states, actions, code_values):
        with torch.no_grad():
            if self.latent_code['type'] == 'categorical':
                logits = self.predict_q(states, actions)
                logp = torch.log_softmax(logits, dim=-1)
                targets = torch.argmax(code_values, dim=1)

                # log q(c|f)
                rewards_us = logp.gather(1, targets.view(-1, 1)) 

                # 计算分类准确率
                preds = torch.argmax(logits, dim=1)  # 预测类别
                correct = (preds == targets).float()
                accuracy = correct.mean().item()  # 平均准确率 (float)

                return rewards_us, accuracy

            elif self.latent_code['type'] == 'continuous':
                # use q_head prediction [mu, log_var]
                q_values = self.predict_q(states, actions)
                mu, log_var = q_values.chunk(2, dim=-1)
                # clamp log_var similar to loss for numerical stability
                log_var = torch.clamp(log_var, min=math.log(0.05**2), max=math.log(1.0**2))
                var = log_var.exp()
                # 高斯 log-prob 奖励
                log_q = -0.5 * ((code_values - mu) ** 2 / var + log_var + math.log(2 * math.pi))
                rewards_us = log_q

                return rewards_us



# SGI_GraphQ_VIB: 引入 VIB 层的 GNN 架构
class SGI_GraphQ_VIB(nn.Module):
    def __init__(self, in_channels, action_shape, latent_code, goal_shape=2, final_mlp_hidden_width=64, global_graph_width=32, vib_latent_dim=32):
        super(SGI_GraphQ_VIB, self).__init__()

        # 1. Base Encoder (GNN)
        self.graph_model = HGNN_Q_ENCODER(in_channels, action_shape[0], final_mlp_hidden_width=final_mlp_hidden_width)
        
        # GNN输出的维度 
        self.input_dim = global_graph_width + goal_shape + action_shape[0]

        # 2. VIB Layer (核心修改)
        # 将 GNN 的确定性输出转化为概率分布的参数 mu 和 log_var
        self.vib_mu = nn.Linear(self.input_dim, vib_latent_dim)
        self.vib_logvar = nn.Linear(self.input_dim, vib_latent_dim)
        
        self.vib_latent_dim = vib_latent_dim

        # 3. Heads
        if latent_code['type'] == 'categorical':
            code_dim = latent_code['dim']
        elif latent_code['type'] == 'continuous':
            code_dim = 2
        self.latent_code = latent_code
        
        # Q Head (Classification): 输入变成了 VIB 的隐变量 z
        self.q_head = nn.Linear(vib_latent_dim, code_dim)

        # Feature Head (Auxiliary): 输入也是 VIB 的隐变量 z
        self.feature_head = nn.Linear(vib_latent_dim, latent_code['feature_dim'])

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick: z = mu + sigma * epsilon
        这样梯度可以反向传播经过采样过程
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, state, action):
        """
        返回: 
        - z: 采样后的隐变量 (用于训练)
        - mu: 均值 (用于推理/Eval，或者计算 KL loss)
        - logvar: 对数方差 (用于计算 KL loss)
        """
        # GNN 提取特征
        h = self.graph_model(state, action) 
        
        # 映射到概率分布参数
        mu = self.vib_mu(h)
        logvar = self.vib_logvar(h)
        
        # 采样
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    # -----------------------------------------------------------
    # Feature Branch (Auxiliary Task)
    # -----------------------------------------------------------
    def predict_feature(self, state, action, deterministic=False):
        z, mu, logvar = self.forward(state, action)
        # 如果是评估模式或者计算 Reward，通常建议直接用 mu (确定性)，减少方差
        # 如果是训练模式，必须用 z (随机性)，以实现正则化
        input_feat = mu if deterministic else z
        return self.feature_head(input_feat), mu, logvar

    def calculate_feature_loss(self, states, actions, target_features, beta=1e-3):
        """
        Loss = MSE(Reconstruction) + Beta * KL_Divergence
        """
        pred_feat, mu, logvar = self.predict_feature(states, actions, deterministic=False)
        
        # 1. Reconstruction Loss (MSE)
        recon_loss = F.mse_loss(pred_feat, target_features)
        
        # 2. KL Divergence Loss
        # KL(N(mu, sigma^2) || N(0, 1)) = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

        # Beta 是一个超参数，用来权衡“预测准确度”和“信息压缩程度”
        total_loss = recon_loss + beta * kl_loss
        
        return total_loss, recon_loss, kl_loss

    # -----------------------------------------------------------
    # Q Branch (Classification Task)
    # -----------------------------------------------------------
    def predict_q(self, state, action, deterministic=True):
        # 注意：在计算 InfoGAIL Reward 时，建议使用 deterministic=True (即只用 mu)
        # 这样给 PPO 的 Reward 信号更稳定，不会忽高忽低
        z, mu, _ = self.forward(state, action)
        input_q = mu if deterministic else z
        q_values = self.q_head(input_q)
        return q_values

    def calculate_loss(self, states, actions, code_values):
        # 训练 Q 分类头时，我们可以选择是否继续由梯度更新 VIB Encoder
        # 通常建议：让 Q head 也能微调 encoder，但使用 reparameterized z 增加鲁棒性
        q_values = self.predict_q(states, actions, deterministic=False)

        if self.latent_code['type'] == 'categorical':
            loss = F.cross_entropy(q_values, torch.argmax(code_values, dim=1))
        elif self.latent_code['type'] == 'continuous':
            # ... (保持你原有的连续逻辑不变) ...
            mu_q, log_var_q = q_values.chunk(2, dim=-1)
            # ...
            assert 1 == 0 , "continuous code is not supported now."

        return loss

    # -----------------------------------------------------------
    # Reward Calculation (Inference)
    # -----------------------------------------------------------
    def calculate_reward(self, states, actions, code_values):
        with torch.no_grad():
            if self.latent_code['type'] == 'categorical':
                logits = self.predict_q(states, actions, deterministic=True)
                logp = torch.log_softmax(logits, dim=-1)
                targets = torch.argmax(code_values, dim=1)

                # log q(c|f)
                rewards_us = logp.gather(1, targets.view(-1, 1)) 

                # 计算分类准确率
                preds = torch.argmax(logits, dim=1)  # 预测类别
                correct = (preds == targets).float()
                accuracy = correct.mean().item()  # 平均准确率 (float)

                return rewards_us, accuracy

            elif self.latent_code['type'] == 'continuous':
                assert 1 == 0, "continuous code is not supported now."






































## NEW CODE after 10.14
class FeatureQ(nn.Module):
    def __init__(self, latent_code, action_shape, final_mlp_hidden_width=64):
        super(FeatureQ, self).__init__()

        self.latent_code = latent_code
        feature_dim = latent_code['feature_dim']
        ## concatenate features and actions or not
        self.only_features = False  # True / False
        if self.only_features:
            assert False, "only_features=True is not supported now."
            input_dim = feature_dim
        else:
            input_dim = feature_dim + action_shape[0]

        if latent_code['type'] == 'categorical':
            code_dim = latent_code['dim']
        elif latent_code['type'] == 'continuous':
            # 输出 mu 和 log_var
            code_dim = 2

        self.model = FinalPredMLP(input_dim, code_dim, final_mlp_hidden_width)

    def forward(self, features, actions):
        ## concatenate features and actions or not
        if self.only_features:
            x = features
        else:
            x = torch.cat([features, actions], dim=-1)

        out = self.model(x)
        if self.latent_code['type'] == 'categorical':
            return out  # logits
        elif self.latent_code['type'] == 'continuous':
            mu, log_var = out.chunk(2, dim=-1)
            # 限制 log_var 避免数值爆炸/塌缩
            log_var = torch.clamp(log_var, min=math.log(0.05**2), max=math.log(1.0**2))
            return mu, log_var

    def calculate_loss(self, features, actions, code_values):
        if self.latent_code['type'] == 'categorical':
            logits = self.forward(features, actions)
            loss = F.cross_entropy(logits, torch.argmax(code_values, dim=1))
        elif self.latent_code['type'] == 'continuous':
            mu, log_var = self.forward(features, actions)
            var = log_var.exp()
            # 负对数似然
            nll = 0.5 * ((code_values - mu) ** 2 / var + log_var + math.log(2 * math.pi))
            loss = nll.mean()
        return loss

    def calculate_reward(self, features, actions, code_values):
        with torch.no_grad():
            if self.latent_code['type'] == 'categorical':
                logits = self.forward(features, actions)
                logp = torch.log_softmax(logits, dim=-1)
                targets = torch.argmax(code_values, dim=1)

                # log q(c|f)
                rewards_us = logp.gather(1, targets.view(-1, 1)) 

                # 计算分类准确率
                preds = torch.argmax(logits, dim=1)  # 预测类别
                correct = (preds == targets).float()
                accuracy = correct.mean().item()  # 平均准确率 (float)

                return rewards_us, accuracy

            elif self.latent_code['type'] == 'continuous':
                mu, log_var = self.forward(features, actions)
                var = log_var.exp()
                # 高斯 log-prob 奖励
                log_q = -0.5 * ((code_values - mu) ** 2 / var + log_var + math.log(2 * math.pi))
                rewards_us = log_q

                return rewards_us







## OLD CODE
class FeatureQ_old(nn.Module):
    def __init__(self, latent_code, action_shape, final_mlp_hidden_width=64):
        super(FeatureQ, self).__init__()

        self.latent_code = latent_code
        feature_dim = latent_code['feature_dim']
        if latent_code['type'] == 'categorical':
            code_dim = latent_code['dim']
        elif latent_code['type'] == 'continuous':
            code_dim = 1
        self.model = FinalPredMLP(feature_dim + action_shape[0], code_dim, final_mlp_hidden_width)

        # self.reward_us_coef = 1.0 # 0.1 # 1.0
        

    def forward(self, features, actions):
            q = self.model(torch.cat([features, actions], dim=-1))
            return q
    
    def calculate_loss(self, features, actions, code_values):
        # Calculate the loss for training the Q function.
        if self.latent_code['type'] == 'categorical':
            loss = F.cross_entropy(self.forward(features, actions), torch.argmax(code_values, dim=1))
        elif self.latent_code['type'] == 'continuous':
            loss = F.mse_loss(self.forward(features, actions), code_values)

        return loss
    
    def calculate_reward(self, features, actions, code_values):
        with torch.no_grad():
            q = self.forward(features, actions)
            if self.latent_code['type'] == 'categorical':
                rewards_us = (-F.cross_entropy(q, torch.argmax(code_values, dim=1), reduction='none')).view(-1,1)
            elif self.latent_code['type'] == 'continuous':
                rewards_us = (-F.mse_loss(q, code_values, reduction='none')).view(-1,1)

            return rewards_us
        




















        







class GraphQ(nn.Module):
    def __init__(self, in_channels, action_shape, latent_code, goal_shape=2, final_mlp_hidden_width=64, global_graph_width=32):
        super(GraphQ, self).__init__()
        self.latent_code = latent_code
        self.graph_model = HGNN_Q_ENCODER(in_channels, action_shape[0], final_mlp_hidden_width=final_mlp_hidden_width)

        if latent_code['type'] == 'categorical':
            code_dim = latent_code['dim']
        elif latent_code['type'] == 'continuous':
            code_dim = 2

        self.q_head = nn.Linear(global_graph_width + goal_shape + action_shape[0], code_dim)

    def forward(self, state, action):
        x = self.graph_model(state, action)
        return x

    def predict_q(self, state, action):
        x = self.forward(state, action)
        q_values = self.q_head(x)
        return q_values

    def calculate_loss(self, states, actions, code_values):
        q_values = self.predict_q(states, actions)
        if self.latent_code['type'] == 'categorical':
            loss = F.cross_entropy(q_values, torch.argmax(code_values, dim=1))
        elif self.latent_code['type'] == 'continuous':
            mu, log_var = q_values.chunk(2, dim=-1)
            log_var = torch.clamp(log_var, min=math.log(0.05**2), max=math.log(1.0**2))
            var = log_var.exp()
            loss = 0.5 * ((code_values - mu) ** 2 / var + log_var).mean()
        return loss


    def calculate_reward(self, states, actions, code_values):
        with torch.no_grad():
            if self.latent_code['type'] == 'categorical':
                logits = self.predict_q(states, actions)
                logp = torch.log_softmax(logits, dim=-1)
                targets = torch.argmax(code_values, dim=1)
                rewards_us = logp.gather(1, targets.view(-1, 1))
                preds = torch.argmax(logits, dim=1)
                accuracy = (preds == targets).float().mean().item()
                return rewards_us, accuracy
            elif self.latent_code['type'] == 'continuous':
                q_values = self.predict_q(states, actions)
                mu, log_var = q_values.chunk(2, dim=-1)
                log_var = torch.clamp(log_var, min=math.log(0.05**2), max=math.log(1.0**2))
                var = log_var.exp()
                log_q = -0.5 * ((code_values - mu) ** 2 / var + log_var + math.log(2 * math.pi))
                rewards_us = log_q
                return rewards_us

    






class AIRLDiscrim(nn.Module):

    def __init__(self, state_shape, gamma,
                 hidden_units_r=(64, 64),
                 hidden_units_v=(64, 64),
                 hidden_activation_r=nn.ReLU(inplace=True),
                 hidden_activation_v=nn.ReLU(inplace=True)):
        super().__init__()

        self.g = build_mlp(
            input_dim=state_shape[0],
            output_dim=1,
            hidden_units=hidden_units_r,
            hidden_activation=hidden_activation_r
        )
        self.h = build_mlp(
            input_dim=state_shape[0],
            output_dim=1,
            hidden_units=hidden_units_v,
            hidden_activation=hidden_activation_v
        )

        self.gamma = gamma

    def f(self, states, dones, next_states):
        rs = self.g(states)
        vs = self.h(states)
        next_vs = self.h(next_states)
        return rs + self.gamma * (1 - dones) * next_vs - vs

    def forward(self, states, dones, log_pis, next_states):
        # Discriminator's output is sigmoid(f - log_pi).
        return self.f(states, dones, next_states) - log_pis

    def calculate_reward(self, states, dones, log_pis, next_states):
        with torch.no_grad():
            logits = self.forward(states, dones, log_pis, next_states)
            return -F.logsigmoid(-logits)

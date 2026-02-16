import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.data import Batch
import numpy as np

from .da_ppo import DA_PPO
from core.network import GAILDiscrim, GraphDiscrim, GraphQ, GraphDiscrim_Info, SGI_GraphQ_VIB, GraphStateIndependentPolicy_Info_DA_FixedStd
from core.utils import separate_latent_cs
from core.buffer import RolloutBuffer_for_Latent_Feature


class SA_SOCIAL_GAIL(DA_PPO):

    def __init__(self, args, buffer_exp, graph_feature_channels, state_shape, action_shape, latent_code, device, seed,
                 gamma=0.995, rollout_length=2048, mix_buffer=1,
                 batch_size=128, lr_actor=3e-4, lr_critic=3e-4, lr_disc=1e-4, lr_q=3e-4, # 3e-4
                 units_actor=64, units_critic=64,
                 units_disc=64, epoch_ppo=40, epoch_disc=1, epoch_q=40,
                 clip_eps=0.2, lambd=0.97, coef_ent=0.0, max_grad_norm=10.0):     # coef_ent=0.0
        super().__init__(
            graph_feature_channels, state_shape, action_shape, latent_code, device, seed, gamma, rollout_length,
            mix_buffer, lr_actor, lr_critic, units_actor, units_critic,
            epoch_ppo, clip_eps, lambd, coef_ent, max_grad_norm
        )

        # cover the actor in DA_PPO
        self.actor = GraphStateIndependentPolicy_Info_DA_FixedStd(
            in_channels=graph_feature_channels,
            latent_code=latent_code,
            action_shape=action_shape, 
            final_mlp_hidden_width=units_actor
        ).to(device)

        # Expert's buffer.
        self.buffer_exp = buffer_exp

        # init discriminator
        self.disc = GraphDiscrim_Info(
            in_channels = graph_feature_channels,
            action_shape = action_shape,
            latent_code = latent_code,
            final_mlp_hidden_width = units_disc
        ).to(device)
        # discriminator's optimizer                  
        self.optim_disc = Adam(self.disc.parameters(), lr=lr_disc)
        print(f'Now learning rate of D: {lr_disc}')


        # init q_nets and optimizers
        self.action_shape = action_shape

        self.lr_q = lr_q
        self.lr_f = 3e-4

        self.graph_feature_channels = graph_feature_channels
        self.q_nets = self._init_q_nets_optimizers()


        self.learning_steps_disc = 0
        self.learning_steps_q = 0
        self.learning_steps_f = 0
        self.lr_disc = lr_disc


        self.batch_size = batch_size
        self.epoch_disc = epoch_disc

        self.epoch_q = epoch_q
        self.epoch_f = 40 # epoch_q  # set to the same as q


        self.device = device
        self.rollout_length = rollout_length
        self.reward_us_coef = args.reward_us_coef


        ## infoGAIL specific
        # self.q_counter_flag = False  # True

        # if self.q_counter_flag:
        #     self.update_q_flag = False
        #     self.update_q_counter = 50*self.epoch_q # 150*self.epoch_q
        #     print(f'Using q_counter: {self.update_q_counter//self.epoch_q}')
        # else:
        #     print(f'Not using q_counter!')


        ## q update flag
        self.q_update_flag = True  # True / True
        print(f'Now q_update_flag: {self.q_update_flag}')

        ## reward_us_coef_flag
        self.reward_us_coef_flag = False
        print(f'Now reward_us_coef_flag: {self.reward_us_coef_flag}')


        # init latent_buffer
        self.latent_buffer = RolloutBuffer_for_Latent_Feature(latent_code=self.latent_code, device=device, buffer_size=self.rollout_length)
        # self.last_latent_features = self._init_latent_features()

        # for critic da
        self.auto_balancing = args.auto_balancing
        print(f'Now update critic da: {self.auto_balancing}')



    def _init_q_nets_optimizers(self):
        # q_nets as a dict
        # with net and optimizer
        q_nets = {}
        for k,v in self.latent_code.items():
            # init network
            net = SGI_GraphQ_VIB(
                in_channels = self.graph_feature_channels,
                action_shape = self.action_shape,
                latent_code = v,
            ).to(self.device)

            # optimizer_f contrains the graph_model and feature head
            # optimizer_q contrains the Q head
            optimizer_f = Adam(
                list(net.graph_model.parameters()) + list(net.feature_head.parameters()) + 
                list(net.vib_mu.parameters()) + list(net.vib_logvar.parameters()),
                lr=self.lr_f
            )
            optimizer_q = Adam(net.q_head.parameters(), lr=self.lr_q)
            


            q_nets[k] = {'net': net, 'optimizer_f': optimizer_f, 'optimizer_q': optimizer_q}  # 
        return q_nets

    # def _init_latent_features(self):
    #     ext_features = {}
    #     for k, v in self.latent_code.items():
    #         ext_features[k] = np.zeros(v['store_dim'])
    #     return ext_features








    # override the step function
    def step(self, env, state, t, step):
        t += 1

        action, log_pi = self.explore(state, self.latent_code_value)

        # store latent features
        self.latent_buffer.append(env.get_latent_features(action)) # action: range (-1,1)

        next_state, reward, done, _ = env.step(action)
        mask = False if t == env._max_episode_steps else done


        self.buffer.append(state, self.latent_code_value, action, reward, mask, log_pi, next_state)

        assert self.latent_buffer.get_p() == self.buffer.get_p() # ensure aligned

        if done:
            t = 0
            next_state = env.reset()
            self.latent_code_value = self.sample_latent_code_value()


        return next_state, t












    def update(self, writer):
        self.learning_steps += 1

        for _ in range(self.epoch_disc):
            self.learning_steps_disc += 1

            # Samples from current policy's trajectories.
            states, _, actions, _, _ = self.buffer.sample(self.batch_size)[:5]
            states = Batch.from_data_list(states).to(self.device)
            # Samples from expert's demonstrations.
            states_exp, actions_exp = self.buffer_exp.sample(self.batch_size)[:2]
            states_exp = Batch.from_data_list(states_exp).to(self.device)
            # Update discriminator.
            self.update_disc(states, actions, states_exp, actions_exp, writer)



        # First train feature encoder + feature head so GNN learns good features
        for _ in range(self.epoch_f):
            self.learning_steps_f += 1

            # Samples from current policy's trajectories.
            start = np.random.randint(low=0, high=self.rollout_length-self.batch_size,size=1)[0]
            idxes = slice(start, start + self.batch_size)
            
            states_f, _, actions_f, _, _ = self.buffer.sample(self.batch_size, idxes)[:5]
            states_f = Batch.from_data_list(states_f).to(self.device)

            latent_features = self.latent_buffer.sample(self.batch_size, idxes)

            # Update feature part
            self.update_feature(states_f, actions_f, latent_features, writer)



        for _ in range(self.epoch_q):
            self.learning_steps_q += 1

            # Samples from current policy's trajectories.
            states, latent_cs, actions, _, dones = self.buffer.sample(self.batch_size)[:5]
            states = Batch.from_data_list(states).to(self.device)
            # Update q using encoder outputs (classification)
            self.update_q(states, latent_cs, actions, dones, writer)
        

        # We don't use reward signals here,
        states, latent_cs, actions, _, dones, log_pis, next_states = self.buffer.get()
        states = Batch.from_data_list(states).to(self.device)
        next_states = Batch.from_data_list(next_states).to(self.device)

        # Calculate rewards.
        rewards_i = self.disc.calculate_reward(states, actions)
        rewards_us = self.calculate_unsupervised_rewards(states, latent_cs, actions, writer)
        assert rewards_i.shape[0] == rewards_us.shape[0] == self.rollout_length

        rewards = rewards_i + self.reward_us_coef * rewards_us

        with torch.no_grad():
            pos_reward_rate =  (rewards > 0).float().mean().item()
            rewards_us_mean = rewards_us.mean().item()
            rewards_i_mean = rewards_i.mean().item()
        writer.add_scalar('rewards/pos_reward_rate', pos_reward_rate, self.learning_steps)
        writer.add_scalar('rewards/rewards_i', rewards_i_mean, self.learning_steps)
        writer.add_scalar('rewards/rewards_us', rewards_us_mean, self.learning_steps)

        # Update PPO using estimated rewards.
        # self.update_ppo(
        #     states, latent_cs, actions, rewards, dones, log_pis, next_states, writer)
        self.update_ppo_da(
            states, latent_cs, actions, rewards_i, rewards_us, dones, log_pis, next_states, writer
        )


    # seprate the update
    def update_disc(self, states, actions, states_exp, actions_exp, writer):
        ## Output of discriminator is (-inf, inf), not [0, 1].
        # logits_pi, pred_cs = self.disc(states, actions)
        # logits_exp, _ = self.disc(states_exp, actions_exp)
        logits_pi = self.disc(states, actions)
        logits_exp = self.disc(states_exp, actions_exp)

        ## Discriminator is to maximize E_{\pi} [log(1 - D)] + E_{exp} [log(D)].
        alpha = 0.0

        loss_pi = -F.logsigmoid(-logits_pi).mean()
        loss_exp = -F.logsigmoid(logits_exp).mean()
        loss_disc = (1+alpha)*loss_exp + (1-alpha)*loss_pi
        # loss_pi = logits_pi.mean()
        # loss_exp = logits_exp.mean()
        # loss_disc = loss_pi - loss_exp


        self.optim_disc.zero_grad()
        loss_disc.backward()
        self.optim_disc.step()


        if self.learning_steps_disc % self.epoch_disc == 0:
            writer.add_scalar(
                'loss/disc', loss_disc.item(), self.learning_steps)

            # Discriminator's accuracies.
            with torch.no_grad():
                acc_pi = (logits_pi < 0).float().mean().item()
                acc_exp = (logits_exp > 0).float().mean().item()
            writer.add_scalar('stats/acc_pi', acc_pi, self.learning_steps)
            writer.add_scalar('stats/acc_exp', acc_exp, self.learning_steps)





    def update_q(self, states, latent_cs, actions, dones, writer):
        ## update q_nets

        # devide latent_cs into different latent codes
        # dict
        separated_latent_cs = separate_latent_cs(latent_cs, self.latent_code)
        for k, v in self.q_nets.items():
            q_net = v['net']
            optim_q = v['optimizer_q']

            # get the latent code for this q_net
            latent_c = separated_latent_cs[k]

            # update q_net
            if self.q_update_flag:
                q_net.train()

                loss_q = q_net.calculate_loss(states, actions, latent_c)

                optim_q.zero_grad()
                loss_q.backward()
                optim_q.step()
            else:
                q_net.eval()
                with torch.no_grad():
                    # do not detach states here; q_net already runs no-grad
                    loss_q = q_net.calculate_loss(states, actions, latent_c)


            if self.learning_steps_q % self.epoch_q == 0:
                writer.add_scalar(
                    f'loss_q/loss_{k}', loss_q.item(), self.learning_steps)

    def update_feature(self, states, actions, latent_features, writer):
        # Train graph encoder + feature head to predict handcrafted features
        extracted_inputs = self.extract_features(latent_features)

        # VIB 的超参数 Beta
        # Beta 越大，正则化越强（压缩得越厉害），可能会降低预测精度但提升泛化性
        # 建议从 1e-4 到 1e-2 之间尝试
        beta = 1e-4

        for k, v in self.q_nets.items():
            q_net = v['net']
            optim_f = v['optimizer_f']

            target_feat = extracted_inputs.get(k, None)
            if target_feat is None:
                continue

            q_net.train()
            
            # 使用新的 Loss 计算函数
            loss_total, loss_recon, loss_kl = q_net.calculate_feature_loss(states, actions, target_feat, beta=beta)

            optim_f.zero_grad()
            loss_total.backward()
            optim_f.step()


            if self.learning_steps_f % self.epoch_f == 0:
                # 记录详细 Loss 以便 Debug
                writer.add_scalar(f'loss_f/total_{k}', loss_total.item(), self.learning_steps)
                writer.add_scalar(f'loss_f/recon_{k}', loss_recon.item(), self.learning_steps)
                writer.add_scalar(f'loss_f/kl_{k}', loss_kl.item(), self.learning_steps)


    def extract_features(self, latent_features):

        ## define each feature extractor
        def extra_speed(v):
            return v
        
        # def extra_nn_dist(v):
        #     # v = [:, 1] of nn_dist
        #     # expand to [:, 3] with before and after value
        #     # replicate padding requires 3D/4D input
        #     x_pad = F.pad(v.unsqueeze(0).unsqueeze(0), (0, 0, 1, 1), mode='replicate').squeeze(0).squeeze(0)
        #     input = torch.cat([x_pad[:-2], x_pad[1:-1], x_pad[2:]], dim=1)
        #     return input
        
        # def extra_nn_dist(v):
        #     return v
        
        # def extra_nn_dist(v, eps=1e-5):
        #     """
        #     将原始 nn-dist 转化为 5步相对变化率特征: (d_t - d_{t-1}) / d_t
        #     输入 v shape: (L, 1)
        #     输出 shape: (L, 5) 每一行代表当前帧及其前后邻域的动态反馈倾向
        #     """
        #     # 1. 计算全局的相对差值序列 (L, 1)
        #     # diff[i] = v[i] - v[i-1]
        #     # 为了保持长度一致，我们在首部做一个 replicate 填充
        #     v_shifted = F.pad(v.unsqueeze(0).unsqueeze(0), (0, 0, 1, 0), mode='replicate').squeeze(0).squeeze(0)
        #     # 计算相对变化率: (v_curr - v_prev) / v_curr
        #     rel_diff = (v_shifted[1:] - v_shifted[:-1]) / (v_shifted[1:] + eps)

        #     # 2. 构造 5 步滑动窗口拼接
        #     # 我们需要前后各扩展 2 帧，总共 5 帧的变化率
        #     # 对 rel_diff 进行 padding (前后各加 2 帧)
        #     rd_pad = F.pad(rel_diff.unsqueeze(0).unsqueeze(0), (0, 0, 2, 2), mode='replicate').squeeze(0).squeeze(0)
            
        #     # 拼接: [t-2, t-1, t, t+1, t+2]
        #     # rd_pad 的长度是 L+4
        #     input = torch.cat([
        #         rd_pad[:-4],     # t-2
        #         rd_pad[1:-3],    # t-1
        #         rd_pad[2:-2],    # t (当前时刻的变化率)
        #         rd_pad[3:-1],    # t+1
        #         rd_pad[4:]       # t+2
        #     ], dim=1)
        #     return input

        def extra_nn_dist(v, eps=1e-5):
            """
            将原始 nn-dist 转化为 5步相对变化率特征: (d_t - d_{t-1}) / d_t
            输入 v shape: (L, 1)
            输出 shape: (L, 5) 每一行代表当前帧及其前后邻域的动态反馈倾向
            """
            # 1. 计算全局的相对差值序列 (L, 1)
            # diff[i] = v[i] - v[i-1]
            # 为了保持长度一致，我们在首部做一个 replicate 填充
            v_shifted = F.pad(v.unsqueeze(0).unsqueeze(0), (0, 0, 1, 0), mode='replicate').squeeze(0).squeeze(0)
            # 计算相对变化率: (v_curr - v_prev) / v_curr
            rel_diff = (v_shifted[1:] - v_shifted[:-1]) / (v_shifted[1:] + eps)
            
            return rel_diff


        def extra_social_comfort(v):
            return v

        def extra_social_gradient(v):
            return v
                        


        # extract inputs from states and actions
        ext_inputs = {}
        for k, v in latent_features.items():
            # for 'speed'
            if k == 'speed':
                input = extra_speed(v)
                ext_inputs[k] = input
                assert input.shape[-1] == self.latent_code[k]['feature_dim']
            elif k == 'nn_dist':
                input = extra_nn_dist(v)
                ext_inputs[k] = input
                assert input.shape[-1] == self.latent_code[k]['feature_dim']
            elif k == 'social_comfort':
                input = extra_social_comfort(v)
                ext_inputs[k] = input
                assert input.shape[-1] == self.latent_code[k]['feature_dim']
            elif k == 'social_gradient':
                input = extra_social_gradient(v)
                ext_inputs[k] = input
                assert input.shape[-1] == self.latent_code[k]['feature_dim']
            else:
                # not finished
                pass

        return ext_inputs




    # cover
    def mormalized_union_gaes(self, gaes_main, gaes_us):
        # if has attribute of reward_us_coef, then use it
        if hasattr(self, 'reward_us_coef') and self.reward_us_coef_flag:
            gaes = gaes_main + self.reward_us_coef * gaes_us
        else:
            gaes = gaes_main
        gaes = (gaes - gaes.mean()) / (gaes.std() + 1e-8)
        return gaes


    def calculate_unsupervised_rewards(self, states, latent_cs, actions, writer):
        ## calculate rewards using q_nets
        rewards = []
        separated_latent_cs = separate_latent_cs(latent_cs, self.latent_code)
        for k, v in self.q_nets.items():
            q_net = v['net']
            # get the latent code for this q_net
            latent_c = separated_latent_cs[k]

            if self.latent_code[k]['type'] == 'categorical':
                # get reward from q_net
                reward_c, accruracy = q_net.calculate_reward(states, actions, latent_c)

                # rewards_us plus log(K), special for diversity_aware rl
                reward_c = reward_c + torch.log(torch.tensor(self.latent_code[k]['dim'], dtype=torch.float32, device=self.device))

                # use writer to log accuracy
                writer.add_scalar(f'rewards/acc_{k}', accruracy, self.learning_steps)
            else:
                reward_c = q_net.calculate_reward(states, actions, latent_c)
            rewards.append(reward_c)

        # sum up the rewards
        rewards_sum = torch.cat(rewards, dim=-1).sum(dim=-1)  # sum over the latent code dimension
        return rewards_sum.unsqueeze(-1)





















    def update_q_si(self, states, latent_cs, actions, dones, writer):
        ## update q_nets

        # devide latent_cs into different latent codes
        # dict
        separated_latent_cs = separate_latent_cs(latent_cs, self.latent_code)
        extracted_features = self.extract_features(states, actions)
        for k, v in self.q_nets.items():
            q_net = v['net']
            optim_q = v['optimizer']

            # get the latent code for this q_net
            latent_c = separated_latent_cs[k]
            features = extracted_features[k]

            # update q_net
            if self.q_update_flag:
                q_net.train()

                loss_q = q_net.calculate_loss(features, actions, latent_c)

                optim_q.zero_grad()
                loss_q.backward()
                optim_q.step()
            else:
                q_net.eval()
                with torch.no_grad():
                    loss_q = q_net.calculate_loss(features.detach(), actions, latent_c)


            if self.learning_steps_q % self.epoch_q == 0:
                writer.add_scalar(
                    f'loss_q/loss_{k}', loss_q.item(), self.learning_steps)

                # Discriminator's accuracies.
                # with torch.no_grad():
                #     judge_cs = torch.argmax(pred_cs, dim=1)
                #     acc_c = (judge_cs == torch.argmax(latent_cs, dim=1)).float().mean().item()
                # writer.add_scalar('stats/acc_cs', acc_c, self.learning_steps)
    



    def calculate_unsupervised_rewards_si(self, states, latent_cs, actions, writer):
        ## calculate rewards using q_nets
        rewards = []
        separated_latent_cs = separate_latent_cs(latent_cs, self.latent_code)
        extracted_features = self.extract_features(states, actions)
        for k, v in self.q_nets.items():
            q_net = v['net']
            # get the latent code for this q_net
            latent_c = separated_latent_cs[k]
            features = extracted_features[k]

            if self.latent_code[k]['type'] == 'categorical':
                # get reward from q_net
                reward_c, accruracy = q_net.calculate_reward(features, actions, latent_c)

                # use writer to log accuracy
                writer.add_scalar(f'rewards/acc_{k}', accruracy, self.learning_steps)
            else:
                reward_c = q_net.calculate_reward(features, actions, latent_c)
            rewards.append(reward_c)

        # sum up the rewards
        rewards_sum = torch.cat(rewards, dim=-1).sum(dim=-1)  # sum over the latent code dimension
        return rewards_sum.unsqueeze(-1)
    

    def update_q_old(self, states, latent_cs, actions, dones, writer):
        ## Output of discriminator is (-inf, inf), not [0, 1].
        pred_cs = self.q(states, actions, dones)

        # q function update
        loss_q = F.cross_entropy(pred_cs, torch.argmax(latent_cs, dim=1))

        if self.q_counter_flag:
            if self.update_q_flag==False and self.learning_steps>100 and self.loss_critic<50:  # self.learning_steps>200 and self.loss_critic<50
                if self.update_q_counter>0:
                    self.update_q_counter -= 1
                else:
                    self.update_q_flag = True
                    writer.add_scalar(
                        'update_q_flag', self.loss_critic, self.learning_steps)

            if self.update_q_flag:    # if self.learning_steps>488:
                self.optim_q.zero_grad()
                loss_q.backward()
                self.optim_q.step()
        else:
            self.optim_q.zero_grad()
            loss_q.backward()
            self.optim_q.step()




        if self.learning_steps_q % self.epoch_q == 0:
            writer.add_scalar(
                'loss/loss_q', loss_q.item(), self.learning_steps)

            # Discriminator's accuracies.
            with torch.no_grad():
                judge_cs = torch.argmax(pred_cs, dim=1)
                acc_c = (judge_cs == torch.argmax(latent_cs, dim=1)).float().mean().item()
            writer.add_scalar('stats/acc_cs', acc_c, self.learning_steps)


    def update_disc_q(self, states, latent_cs, actions, dones, states_exp, actions_exp, writer):
        ## Output of discriminator is (-inf, inf), not [0, 1].
        # logits_pi, pred_cs = self.disc(states, actions)
        # logits_exp, _ = self.disc(states_exp, actions_exp)
        logits_pi, pred_cs = self.disc(states, actions, dones)
        logits_exp, _ = self.disc(states_exp, actions_exp)

        ## Discriminator is to maximize E_{\pi} [log(1 - D)] + E_{exp} [log(D)].
        alpha = 0.0

        loss_pi = -F.logsigmoid(-logits_pi).mean()
        loss_exp = -F.logsigmoid(logits_exp).mean()
        loss_disc = (1+alpha)*loss_exp + (1-alpha)*loss_pi
        # loss_pi = logits_pi.mean()
        # loss_exp = logits_exp.mean()
        # loss_disc = loss_pi - loss_exp


        self.optim_disc.zero_grad()
        loss_disc.backward(retain_graph=True)

        # # gradient_penalty
        # real_data = torch.cat([states_exp, actions_exp], dim=-1)
        # fake_data = torch.cat([states, actions], dim=-1)
        # gradient_penalty = self.calc_gradient_penalty(real_data, fake_data)
        # gradient_penalty.backward()

        self.optim_disc.step()

        # q function update
        loss_q = F.cross_entropy(pred_cs, torch.argmax(latent_cs, dim=1))
        self.optim_q.zero_grad()
        loss_q.backward()
        self.optim_q.step()




        if self.learning_steps_disc % self.epoch_disc == 0:
            writer.add_scalar(
                'loss/disc', loss_disc.item(), self.learning_steps)
            writer.add_scalar(
                'loss/loss_q', loss_q.item(), self.learning_steps)

            # Discriminator's accuracies.
            with torch.no_grad():
                acc_pi = (logits_pi < 0).float().mean().item()
                acc_exp = (logits_exp > 0).float().mean().item()
                judge_cs = torch.argmax(pred_cs, dim=1)
                acc_c = (judge_cs == torch.argmax(latent_cs, dim=1)).float().mean().item()
            writer.add_scalar('stats/acc_pi', acc_pi, self.learning_steps)
            writer.add_scalar('stats/acc_exp', acc_exp, self.learning_steps)
            writer.add_scalar('stats/acc_cs', acc_c, self.learning_steps)



    def calc_gradient_penalty(self, real_data, fake_data):
        alpha = torch.rand(self.batch_size, 1)
        alpha = alpha.expand(real_data.size())
        alpha = alpha.to(self.device)

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        
        interpolates = interpolates.to(self.device)
        interpolates = torch.autograd.Variable(interpolates, requires_grad=True)

        disc_interpolates = self.disc.gp_forward(interpolates)

        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                grad_outputs=torch.ones(disc_interpolates.size()).to(self.device),
                                create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10  # (LAMBDA)
        return gradient_penalty

    def save_models(self, save_dir):
        # add discriminator to the model_state_dict
        # print(f'Save model to {save_dir}.pt')
        model_state_dict = {'actor':self.actor.state_dict(), 'critic':self.critic.state_dict(), 'disc':self.disc.state_dict()}
        torch.save(model_state_dict, save_dir+'.pt')
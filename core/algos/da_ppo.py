import torch
from torch import nn
from torch.optim import Adam
import os
import numpy as np

from .si_base import SI_Algorithm
from core.buffer import RolloutBuffer, SeparatedRolloutBuffer, RolloutBuffer_Info
from core.network import StateIndependentPolicy, StateFunction, GraphStateIndependentPolicy_Info, GraphStateFunction_Info_DA, GraphStateIndependentPolicy_Info_DA
from core.utils import latent_code_dim
from torch_geometric.data import Batch


def calculate_gae(values, rewards, dones, next_values, gamma, lambd, normalize=True):
    # Calculate TD errors.
    deltas = rewards + gamma * next_values * (1 - dones) - values
    # Initialize gae.
    gaes = torch.empty_like(rewards)

    # Calculate gae recursively from behind.
    gaes[-1] = deltas[-1]
    for t in reversed(range(rewards.size(0) - 1)):
        gaes[t] = deltas[t] + gamma * lambd * (1 - dones[t]) * gaes[t + 1]

    target = gaes + values
    
    if normalize:
        gaes = (gaes - gaes.mean()) / (gaes.std() + 1e-8)

    return target, gaes


class DA_PPO(SI_Algorithm):

    def __init__(self, graph_feature_channels, state_shape, action_shape, latent_code, device, seed, gamma=0.995,
                 rollout_length=2048, mix_buffer=1, lr_actor=3e-4,
                 lr_critic=3e-4, units_actor=64, units_critic=64,
                 epoch_ppo=40, clip_eps=0.2, lambd=0.97, coef_ent=0.0,
                 max_grad_norm=10.0):
        super().__init__(state_shape, action_shape, device, seed, gamma)

        # latent code.
        self.latent_code = latent_code
        self.dim_c = latent_code_dim(latent_code)

        # Rollout buffer.
        self.buffer = RolloutBuffer_Info(        # RolloutBuffer
            buffer_size=rollout_length,
            state_shape=state_shape,
            action_shape=action_shape,
            dim_c = self.dim_c,
            device = device,
            mix = mix_buffer
        )
        # self.buffer = SeparatedRolloutBuffer(        
        #     buffer_size=rollout_length,
        #     state_shape=state_shape,
        #     action_shape=action_shape,
        #     device=device,
        #     batch_size=batch_size
        # )

        # Actor.
        # self.actor = StateIndependentPolicy(
        #     state_shape=state_shape,
        #     action_shape=action_shape,
        #     hidden_units=units_actor,
        #     hidden_activation=nn.Tanh()
        # ).to(device)
        print(device)

        self.actor = GraphStateIndependentPolicy_Info_DA(
            in_channels=graph_feature_channels,
            latent_code=latent_code,
            action_shape=action_shape, 
            final_mlp_hidden_width=units_actor
        ).to(device)

        # Critic. multi head for main reward value estimate and auxiliary reward (diversity reward) value estimate
        self.critic = GraphStateFunction_Info_DA(
            in_channels=graph_feature_channels,
            latent_code=latent_code,
            final_mlp_hidden_width=units_critic,
        ).to(device)


        self.optim_actor = Adam(self.actor.parameters(), lr=lr_actor)
        self.optim_critic = Adam(self.critic.parameters(), lr=lr_critic)

        self.learning_steps_ppo = 0
        self.rollout_length = rollout_length
        self.epoch_ppo = epoch_ppo
        self.clip_eps = clip_eps
        self.lambd = lambd
        self.coef_ent = coef_ent
        self.max_grad_norm = max_grad_norm
        self.device = device

        # infoGAIL specific
        self.latent_code_value = self.sample_latent_code_value()

        self.loss_critic = float('inf')

        

    def sample_latent_code_value(self):
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
        latent_code_valute_list = []
        for k,v in self.latent_code.items():
            if v['type'] == 'categorical':
                ## random
                idx = torch.randint(v['dim'], (1,)).item()  # sample from categorical

                ## fixed
                # idx = 0

                code_value = torch.zeros(v['dim'], device=self.device)
                code_value[idx] = 1.0  # one-hot encoding
                latent_code_valute_list.append(code_value)
            elif v['type'] == 'continuous':
                code_value = torch.rand(1, device=self.device) * (v['range'][1] - v['range'][0]) + v['range'][0]
                latent_code_valute_list.append(code_value)
            else:
                raise ValueError(f"Unknown type {v['type']} for latent code {k}")
        return torch.cat(latent_code_valute_list, dim=0).to(self.device)
        


    # def sample_latent_c(self):
    #     # dimansion is dim_c, one hot vector
    #     idx = torch.randint(self.dim_c, (1,)).item()
    #     self.latent_c = torch.zeros(self.dim_c, device=self.device)
    #     self.latent_c[idx] = 1.0




    def is_update(self, step):
        return step % self.rollout_length == 0

    def step(self, env, state, t, step):
        t += 1

        action, log_pi = self.explore(state, self.latent_code_value)
        next_state, reward, done, _ = env.step(action)
        mask = False if t == env._max_episode_steps else done

        self.buffer.append(state, self.latent_code_value, action, reward, mask, log_pi, next_state)

        if done:
            t = 0
            next_state = env.reset()
            self.latent_code_value = self.sample_latent_code_value()

        return next_state, t

    def update(self, writer):
        self.learning_steps += 1

        states, latent_cs, actions, rewards, dones, log_pis, next_states = self.buffer.get()

        states = Batch.from_data_list(states).to(self.device)
        next_states = Batch.from_data_list(next_states).to(self.device)

        # self.update_ppo(
        #     states, latent_cs, actions, rewards, dones, log_pis, next_states, writer)
        self.update_ppo_da(
            states, latent_cs, actions, rewards, torch.zeros_like(rewards), dones, log_pis, next_states, writer
        )






            

    def update_ppo(self, states, latent_cs, actions, rewards, dones, log_pis, next_states,
                   writer):
        with torch.no_grad():
            values = self.critic(states,latent_cs)
            next_values = self.critic(next_states,latent_cs)

        targets, gaes = calculate_gae(
            values, rewards, dones, next_values, self.gamma, self.lambd)

        for _ in range(self.epoch_ppo):
            self.learning_steps_ppo += 1
            self.update_critic(states, latent_cs, targets, writer)
            self.update_actor(states, latent_cs, actions, log_pis, gaes, writer)

    def update_critic(self, states, latent_cs, targets, writer):
        loss_critic = (self.critic(states,latent_cs) - targets).pow_(2).mean()
        self.optim_critic.zero_grad()
        loss_critic.backward(retain_graph=False)
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.optim_critic.step()

        if self.learning_steps_ppo % self.epoch_ppo == 0:
            writer.add_scalar(
                'loss/critic', loss_critic.item(), self.learning_steps)
            
            # update loss_critic
            self.loss_critic = loss_critic.item()

    def update_actor(self, states, latent_cs, actions, log_pis_old, gaes, writer):
        log_pis = self.actor.evaluate_log_pi(states, latent_cs, actions)
        entropy = -log_pis.mean()

        ratios = (log_pis - log_pis_old).exp_()
        loss_actor1 = -ratios * gaes
        loss_actor2 = -torch.clamp(
            ratios,
            1.0 - self.clip_eps,
            1.0 + self.clip_eps
        ) * gaes
        loss_actor = torch.max(loss_actor1, loss_actor2).mean()

        self.optim_actor.zero_grad()
        (loss_actor - self.coef_ent * entropy).backward(retain_graph=False)
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.optim_actor.step()

        if self.learning_steps_ppo % self.epoch_ppo == 0:
            writer.add_scalar(
                'loss/actor', loss_actor.item(), self.learning_steps)
            writer.add_scalar(
                'stats/entropy', entropy.item(), self.learning_steps)





    def mormalized_union_gaes(self, gaes_main, gaes_us):
        # if has attribute of reward_us_coef, then use it
        if hasattr(self, 'reward_us_coef'):
            gaes = gaes_main + self.reward_us_coef * gaes_us
        else:
            gaes = gaes_main
        gaes = (gaes - gaes.mean()) / (gaes.std() + 1e-8)
        return gaes

    # diversity-aware ppo update
    def update_ppo_da(self, states, latent_cs, actions, rewards_main, rewards_us, dones, log_pis, next_states,
                   writer):
        with torch.no_grad():
            values_main, values_us = self.critic(states,latent_cs)
            next_values_main, next_values_us = self.critic(next_states,latent_cs)
            
        targets_main, gaes_main = calculate_gae(
            values_main, rewards_main, dones, next_values_main, self.gamma, self.lambd, normalize=False)
        targets_us, gaes_us = calculate_gae(
            values_us, rewards_us, dones, next_values_us, self.gamma, self.lambd, normalize=False)

        # normalized union gaes
        gaes = self.mormalized_union_gaes(gaes_main, gaes_us)

        for _ in range(self.epoch_ppo):
            self.learning_steps_ppo += 1
            if hasattr(self, 'auto_balancing') and self.auto_balancing:
                self.update_critic_da(states, latent_cs, targets_main, targets_us, writer)
            else:
                self.update_critic_one_head(states, latent_cs, targets_main, writer)
            self.update_actor(states, latent_cs, actions, log_pis, gaes, writer)

    def update_critic_da(self, states, latent_cs, targets_main, targets_us, writer):
        values_main, values_us = self.critic(states,latent_cs)
        loss_critic_main = (values_main - targets_main).pow_(2).mean()
        loss_critic_us = (values_us - targets_us).pow_(2).mean()
        loss_critic = loss_critic_main + loss_critic_us

        self.optim_critic.zero_grad()
        loss_critic.backward(retain_graph=False)
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.optim_critic.step()

        if self.learning_steps_ppo % self.epoch_ppo == 0:
            writer.add_scalar(
                'loss/critic', loss_critic.item(), self.learning_steps)
            writer.add_scalar(
                'loss/critic_main', loss_critic_main.item(), self.learning_steps)
            writer.add_scalar(
                'loss/critic_us', loss_critic_us.item(), self.learning_steps)
            
            # update loss_critic
            self.loss_critic = loss_critic.item()

    def update_critic_one_head(self, states, latent_cs, targets_main, writer):
        values_main, _ = self.critic(states,latent_cs)
        loss_critic_main = (values_main - targets_main).pow_(2).mean()
        loss_critic = loss_critic_main
        
        self.optim_critic.zero_grad()
        loss_critic.backward(retain_graph=False)
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.optim_critic.step()

        if self.learning_steps_ppo % self.epoch_ppo == 0:
            writer.add_scalar(
                'loss/critic', loss_critic.item(), self.learning_steps)
            
            # update loss_critic
            self.loss_critic = loss_critic.item()






    def save_models(self, save_dir):
        # pass
        model_state_dict = {'actor':self.actor.state_dict(), 'critic':self.critic.state_dict()}
        torch.save(model_state_dict, save_dir+'.pt')

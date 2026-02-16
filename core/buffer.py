import os
import numpy as np
import torch
from .network import GraphData


class SerializedBuffer:

    def __init__(self, path, device):
        tmp = torch.load(path)
        self.buffer_size = self._n = tmp['state'].size(0)
        self.device = device

        self.states = tmp['state'].clone().to(self.device)
        self.actions = tmp['action'].clone().to(self.device)
        self.rewards = tmp['reward'].clone().to(self.device)
        self.dones = tmp['done'].clone().to(self.device)
        self.next_states = tmp['next_state'].clone().to(self.device)

    def sample(self, batch_size):
        idxes = np.random.randint(low=0, high=self._n, size=batch_size)
        return (
            self.states[idxes],
            self.actions[idxes],
            self.rewards[idxes],
            self.dones[idxes],
            self.next_states[idxes]
        )

class SerializedBuffer_SA:

    def __init__(self, path, device):
        print(f'Loading expert demo from {path}...')

        tmp = torch.load(path)
        self.buffer_size = self._n = len(tmp['state'])
        self.device = device

        print(f'Expert demo size: {self.buffer_size}')

        self.states = tmp['state']    #.clone().to(self.device)
        self.actions = tmp['action'].clone().to(self.device)

    def sample(self, batch_size):
        idxes = np.random.randint(low=0, high=self._n, size=batch_size)

        states = []
        for i in idxes:
            s = self.states[i]
            states.append(s)

        return (
            states,
            self.actions[idxes]
        )


class Buffer(SerializedBuffer):

    def __init__(self, buffer_size, state_shape, action_shape, device):
        self._n = 0
        self._p = 0
        self.buffer_size = buffer_size
        self.device = device

        self.states = torch.empty(
            (buffer_size, *state_shape), dtype=torch.float, device=device)
        self.actions = torch.empty(
            (buffer_size, *action_shape), dtype=torch.float, device=device)
        self.rewards = torch.empty(
            (buffer_size, 1), dtype=torch.float, device=device)
        self.dones = torch.empty(
            (buffer_size, 1), dtype=torch.float, device=device)
        self.next_states = torch.empty(
            (buffer_size, *state_shape), dtype=torch.float, device=device)

    def append(self, state, action, reward, done, next_state):
        self.states[self._p].copy_(torch.from_numpy(state))
        self.actions[self._p].copy_(torch.from_numpy(action))
        self.rewards[self._p] = float(reward)
        self.dones[self._p] = float(done)
        self.next_states[self._p].copy_(torch.from_numpy(next_state))

        self._p = (self._p + 1) % self.buffer_size
        self._n = min(self._n + 1, self.buffer_size)

    def save(self, path):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))

        torch.save({
            'state': self.states.clone().cpu(),
            'action': self.actions.clone().cpu(),
            'reward': self.rewards.clone().cpu(),
            'done': self.dones.clone().cpu(),
            'next_state': self.next_states.clone().cpu(),
        }, path)




class RolloutBuffer:

    def __init__(self, buffer_size, state_shape, action_shape, device, mix=1):
        self._n = 0
        self._p = 0
        self.mix = mix
        self.buffer_size = buffer_size
        self.total_size = mix * buffer_size
        self.device = device

        self.states = [None for i in range(self.total_size)]
        self.actions = torch.empty(
            (self.total_size, *action_shape), dtype=torch.float, device=device)
        self.rewards = torch.empty(
            (self.total_size, 1), dtype=torch.float, device=device)
        self.dones = torch.empty(
            (self.total_size, 1), dtype=torch.float, device=device)
        self.log_pis = torch.empty(
            (self.total_size, 1), dtype=torch.float, device=device)
        self.next_states = [None for i in range(self.total_size)]
        # self.next_states = torch.empty(
        #     (self.total_size, *state_shape), dtype=torch.float, device=device)

    def append(self, state, action, reward, done, log_pi, next_state):
        self.states[self._p] = state.to(self.device)      #(torch.from_numpy(state))
        self.actions[self._p].copy_(torch.from_numpy(action))
        self.rewards[self._p] = float(reward)
        self.dones[self._p] = float(done)
        self.log_pis[self._p] = float(log_pi)
        self.next_states[self._p] = next_state.to(self.device)       #(torch.from_numpy(next_state))

        self._p = (self._p + 1) % self.total_size
        self._n = min(self._n + 1, self.total_size)

    def get(self):
        assert self._p % self.buffer_size == 0
        start = (self._p - self.buffer_size) % self.total_size
        idxes = slice(start, start + self.buffer_size)
        return (
            self.states[idxes],
            self.actions[idxes],
            self.rewards[idxes],
            self.dones[idxes],
            self.log_pis[idxes],
            self.next_states[idxes]
        )

    def sample(self, batch_size, idxes = None):
        assert self._p % self.buffer_size == 0
        if idxes is None:
            idxes = np.random.randint(low=0, high=self._n, size=batch_size)

        states, next_states = [], []
        for i in idxes:
            s = self.states[i]
            ns = self.next_states[i]
            states.append(s)
            next_states.append(ns)

        return (
            states,
            self.actions[idxes],
            self.rewards[idxes],
            self.dones[idxes],
            self.log_pis[idxes],
            next_states
        )









class RolloutBuffer_Info:

    def __init__(self, buffer_size, state_shape, action_shape, dim_c, device, mix=1):
        self._n = 0
        self._p = 0
        self.mix = mix
        self.buffer_size = buffer_size
        self.total_size = mix * buffer_size
        self.device = device

        self.states = [None for i in range(self.total_size)]
        self.latent_cs = torch.empty(
            (self.total_size, dim_c), dtype=torch.float, device=device)
        self.actions = torch.empty(
            (self.total_size, *action_shape), dtype=torch.float, device=device)
        self.rewards = torch.empty(
            (self.total_size, 1), dtype=torch.float, device=device)
        self.dones = torch.empty(
            (self.total_size, 1), dtype=torch.float, device=device)
        self.log_pis = torch.empty(
            (self.total_size, 1), dtype=torch.float, device=device)
        self.next_states = [None for i in range(self.total_size)]
        # self.next_states = torch.empty(
        #     (self.total_size, *state_shape), dtype=torch.float, device=device)

    def append(self, state, latent_c, action, reward, done, log_pi, next_state):
        self.states[self._p] = state.to(self.device)      #(torch.from_numpy(state))
        self.latent_cs[self._p].copy_(latent_c)
        self.actions[self._p].copy_(torch.from_numpy(action))
        self.rewards[self._p] = float(reward)
        self.dones[self._p] = float(done)
        self.log_pis[self._p] = float(log_pi)
        self.next_states[self._p] = next_state.to(self.device)       #(torch.from_numpy(next_state))

        self._p = (self._p + 1) % self.total_size
        self._n = min(self._n + 1, self.total_size)

    def get(self):
        assert self._p % self.buffer_size == 0
        start = (self._p - self.buffer_size) % self.total_size
        idxes = slice(start, start + self.buffer_size)
        return (
            self.states[idxes],
            self.latent_cs[idxes],
            self.actions[idxes],
            self.rewards[idxes],
            self.dones[idxes],
            self.log_pis[idxes],
            self.next_states[idxes]
        )
    
    def sample(self, batch_size, idxes = None):
        # sample coninuteous data
        if idxes is None:
            start = np.random.randint(low=0, high=self.buffer_size-batch_size,size=1)[0]
            idxes = slice(start, start + batch_size)
        return (
            self.states[idxes],
            self.latent_cs[idxes],
            self.actions[idxes],
            self.rewards[idxes],
            self.dones[idxes],
            self.log_pis[idxes],
            self.next_states[idxes]
        )

    def get_p(self):
        return self._p
    





class RolloutBuffer_for_Latent_Feature:

    def __init__(self, latent_code, device, buffer_size, mix=1):
        self._n = 0
        self._p = 0
        self.mix = mix
        self.buffer_size = buffer_size
        self.total_size = mix * buffer_size
        self.device = device

        self.latent_code = latent_code
        self.latent_features = {}
        for k,v in latent_code.items():
            self.latent_features[k] = torch.empty(
                (self.total_size, v['store_dim']), dtype=torch.float, device=device)

    def append(self, latent_features):
        for k,v in latent_features.items():
            self.latent_features[k][self._p].copy_(torch.from_numpy(v))

        self._p = (self._p + 1) % self.total_size
        self._n = min(self._n + 1, self.total_size)

    def get(self):
        assert self._p % self.buffer_size == 0
        start = (self._p - self.buffer_size) % self.total_size
        idxes = slice(start, start + self.buffer_size)

        latent_features = {}
        for k,v in self.latent_features.items():
            latent_features[k] = v[idxes]

        return (
            latent_features
        )
    
    def sample(self, batch_size, idxes = None):
        if idxes is None:
            # sample coninuteous data
            start = np.random.randint(low=0, high=self.buffer_size-batch_size,size=1)[0]
            idxes = slice(start, start + self.buffer_size)


        latent_features = {}
        for k,v in self.latent_features.items():
            latent_features[k] = v[idxes]

        return (
            latent_features 
        )

    def get_p(self):
        return self._p



# class RolloutBuffer_Info:

#     def __init__(self, buffer_size, state_shape, dim_c, action_shape, device, mix=1):
#         self._n = 0
#         self._p = 0
#         self.mix = mix
#         self.buffer_size = buffer_size
#         self.total_size = mix * buffer_size
#         self.device = device

#         self.states = [None for i in range(self.total_size)]
#         self.latent_cs = torch.empty(
#             (self.total_size, dim_c), dtype=torch.float, device=device)
#         self.actions = torch.empty(
#             (self.total_size, *action_shape), dtype=torch.float, device=device)
#         self.rewards = torch.empty(
#             (self.total_size, 1), dtype=torch.float, device=device)
#         self.dones = torch.empty(
#             (self.total_size, 1), dtype=torch.float, device=device)
#         self.log_pis = torch.empty(
#             (self.total_size, 1), dtype=torch.float, device=device)
#         self.next_states = [None for i in range(self.total_size)]
#         # self.next_states = torch.empty(
#         #     (self.total_size, *state_shape), dtype=torch.float, device=device)

#     def append(self, state, latent_c, action, reward, done, log_pi, next_state):
#         self.states[self._p] = state.to(self.device)      #(torch.from_numpy(state))
#         self.latent_cs[self._p].copy_(latent_c)
#         self.actions[self._p].copy_(torch.from_numpy(action))
#         self.rewards[self._p] = float(reward)
#         self.dones[self._p] = float(done)
#         self.log_pis[self._p] = float(log_pi)
#         self.next_states[self._p] = next_state.to(self.device)       #(torch.from_numpy(next_state))

#         self._p = (self._p + 1) % self.total_size
#         self._n = min(self._n + 1, self.total_size)

#     def get(self):
#         assert self._p % self.buffer_size == 0
#         start = (self._p - self.buffer_size) % self.total_size
#         idxes = slice(start, start + self.buffer_size)
#         return (
#             self.states[idxes],
#             self.latent_cs[idxes],
#             self.actions[idxes],
#             self.rewards[idxes],
#             self.dones[idxes],
#             self.log_pis[idxes],
#             self.next_states[idxes]
#         )
    
#     def sample(self, batch_size):
#         # sample coninuteous data
#         start = np.random.randint(low=0, high=self.buffer_size-batch_size,size=1)[0]
#         idxes = slice(start, start + batch_size)
#         return (
#             self.states[idxes],
#             self.latent_cs[idxes],
#             self.actions[idxes],
#             self.rewards[idxes],
#             self.dones[idxes],
#             self.log_pis[idxes],
#             self.next_states[idxes]
#         )

#     def sample_old(self, batch_size):
#         assert self._p % self.buffer_size == 0
#         idxes = np.random.randint(low=0, high=self._n, size=batch_size)

#         states, next_states = [], []
#         for i in idxes:
#             s = self.states[i]
#             ns = self.next_states[i]
#             states.append(s)
#             next_states.append(ns)

#         return (
#             states,
#             self.latent_cs[idxes],
#             self.actions[idxes],
#             self.rewards[idxes],
#             self.dones[idxes],
#             self.log_pis[idxes],
#             next_states
#         )


    

class SeparatedRolloutBuffer:

    def __init__(self, buffer_size, state_shape, action_shape, device, batch_size):
        self._n = 0
        self._p = 0     
        self.batch_size = batch_size
        self.total_size = buffer_size

        self.states = torch.empty(
            (self.total_size, *state_shape), dtype=torch.float, device=device)
        self.actions = torch.empty(
            (self.total_size, *action_shape), dtype=torch.float, device=device)
        self.rewards = torch.empty(
            (self.total_size, 1), dtype=torch.float, device=device)
        self.dones = torch.empty(
            (self.total_size, 1), dtype=torch.float, device=device)
        self.log_pis = torch.empty(
            (self.total_size, 1), dtype=torch.float, device=device)
        self.next_states = torch.empty(
            (self.total_size, *state_shape), dtype=torch.float, device=device)

    def append(self, state, action, reward, done, log_pi, next_state):
        self.states[self._p].copy_(torch.from_numpy(state))
        self.actions[self._p].copy_(torch.from_numpy(action))
        self.rewards[self._p] = float(reward)
        self.dones[self._p] = float(done)
        self.log_pis[self._p] = float(log_pi)
        self.next_states[self._p].copy_(torch.from_numpy(next_state))

        self._p = (self._p + 1) % self.total_size
        self._n = min(self._n + 1, self.total_size)

    def get(self):
        start = self.total_size-self.batch_size
        idxes = slice(start, self.total_size)
        return (
            self.states[idxes],
            self.actions[idxes],
            self.rewards[idxes],
            self.dones[idxes],
            self.log_pis[idxes],
            self.next_states[idxes]
        )

    def sample(self, batch_size):
        idxes = np.random.randint(low=0, high=self._n-self.batch_size, size=batch_size)
        return (
            self.states[idxes],
            self.actions[idxes],
            self.rewards[idxes],
            self.dones[idxes],
            self.log_pis[idxes],
            self.next_states[idxes]
        )

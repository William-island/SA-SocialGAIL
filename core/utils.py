from tqdm import tqdm
import numpy as np
import torch

from .buffer import Buffer

import argparse
import yaml
from datetime import datetime
from pathlib import Path


def soft_update(target, source, tau):
    for t, s in zip(target.parameters(), source.parameters()):
        t.data.mul_(1.0 - tau)
        t.data.add_(tau * s.data)


def disable_gradient(network):
    for param in network.parameters():
        param.requires_grad = False


def add_random_noise(action, std):
    action += np.random.randn(*action.shape) * std
    return action.clip(-1.0, 1.0)


def collect_demo(env, algo, buffer_size, device, std, p_rand, seed=0):
    env.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    buffer = Buffer(
        buffer_size=buffer_size,
        state_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        device=device
    )

    total_return = 0.0
    num_episodes = 0

    state = env.reset()
    t = 0
    episode_return = 0.0

    for _ in tqdm(range(1, buffer_size + 1)):
        t += 1

        if np.random.rand() < p_rand:
            action = env.action_space.sample()
        else:
            action = algo.exploit(state)
            action = add_random_noise(action, std)

        next_state, reward, done, _ = env.step(action)
        mask = False if t == env._max_episode_steps else done
        buffer.append(state, action, reward, mask, next_state)
        episode_return += reward

        if done:
            num_episodes += 1
            total_return += episode_return
            state = env.reset()
            t = 0
            episode_return = 0.0

        state = next_state

    print(f'Mean return of the expert is {total_return / num_episodes}')
    return buffer













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

def separate_latent_cs(latent_cs, latent_code):
    # separate latent code into categorical and continuous parts 
    # according to the latent code

    separated_latent_cs = {}
    ptr = 0
    for key, value in latent_code.items():
        if value['type'] == 'categorical':
            dim = value['dim']
            separated_latent_cs[key] = latent_cs[:, ptr:ptr + dim]
            ptr += dim
        elif value['type'] == 'continuous':
            separated_latent_cs[key] = latent_cs[:, ptr:ptr + 1]
            ptr += 1
        else:
            raise ValueError(f"Unknown type {value['type']} for latent code {key}")
    return separated_latent_cs
    




def save_args_to_yaml(args, save_dir):
    # 将 Namespace 转换成普通字典
    args_dict = vars(args)

    # # 创建保存目录
    # Path(save_dir).mkdir(parents=True, exist_ok=True)

    # 给配置文件加上时间戳
    # timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    # config_path = Path(save_dir) / f"config_{timestamp}.yaml"
    config_path = Path(save_dir) / f"config.yaml"

    # 保存为 yaml 文件
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(args_dict, f, allow_unicode=True, sort_keys=False)

    print(f"训练参数已保存到 {config_path}")
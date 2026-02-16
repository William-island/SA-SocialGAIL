import gym
from core.crowd_env import CrowdEnv

gym.logger.set_level(40)


def make_env(args, TT_type):
    env = CrowdEnv(args, TT_type)
    return NormalizedEnv(env, scale_factor=args.scale_factor)
    # return NormalizedEnv(env)
    # return NormalizedEnv(gym.make(env_id))


class NormalizedEnv(gym.Wrapper):

    def __init__(self, env, scale_factor):
        gym.Wrapper.__init__(self, env)
        self._max_episode_steps = env._max_episode_steps

        # self.scale = env.action_space.high
        self.scale = scale_factor
        print(f'Environment scale factor: {self.scale}')

    def step(self, action):
        return self.env.step(action * self.scale)

from unityagents import UnityEnvironment

import numpy as np
import gym
from gym import spaces

class ReacherEnv:
    def __init__(self, env):
        self.env = env
        self.brain_name = self.env.brain_names[0]
        self.brain = self.env.brains[self.brain_name]

        self.action_dim = self.brain.vector_action_space_size
        self.action_space = spaces.Box(low= np.array([-1.0]*4),
                                       high=np.array([1.0]*4),
                                       dtype=np.float32)
        state = self.reset()
        self.state_dim = state.shape[1]
        self.episode_rewards = []


    def reset(self):
        #print('environment reset')
        env_info = self.env.reset(train_mode=True)[self.brain_name]
        self.state = env_info.vector_observations
        return self.state


    def step(self, actions):
        actions = np.clip(actions, self.action_space.low, self.action_space.high)
        env_info = self.env.step(actions)[self.brain_name]
        next_states = env_info.vector_observations
        rewards = np.array(env_info.rewards)
        dones = np.array(env_info.local_done)
        infos = tuple([{} for i in range(20)])
        #print(dones)
        return next_states, rewards, dones, infos

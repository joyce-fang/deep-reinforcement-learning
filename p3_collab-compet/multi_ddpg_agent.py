import numpy as np
import random


from model import Actor, Critic
from noise import OUNoise
from buffer import ReplayBuffer

import torch
import torch.nn.functional as F
import torch.optim as optim

device = 'cpu'


class MultiAgent:
    def __init__(self, config):

        self.random_seeds = config['random_seeds']
        self.params = config['params']
        self.memory = ReplayBuffer(self.params['action_size'],
                                   self.params['buffer_size'],
                                   self.params['batch_size'],
                                   device,
                                   self.random_seeds[0])
        self.params['memory'] = self.memory

        self.ddpg_agents = [Agent(self.params, self.random_seeds[i]) for i in range(2)]

        self.t_step = 0

    def act(self, states):
        actions = [agent.act(np.expand_dims(state, axis=0)) for agent, state in zip(self.ddpg_agents, states)]
        #actions = [agent.act(states) for agent in self.ddpg_agents]
        return actions

    def step(self, states, actions, rewards, next_states, dones):
        self.t_step += 1

        for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
            self.memory.add(state, action, reward, next_state, done)

        if (len(self.memory) > self.params['batch_size']) and (self.t_step % self.params['num_steps_per_update'] == 0):
            for agent in self.ddpg_agents:
                experiences = self.memory.sample()
                agent.learn(experiences, self.params['gamma'])

    def reset(self):
        for agent in self.ddpg_agents:
            agent.reset()



class Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, params, random_seed):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.n_agents = 1
        self.state_size = params['state_size']
        self.action_size = params['action_size']
        self.batch_size = params['batch_size']
        self.gamma = params['gamma']
        self.tau = params['tau']
        self.seed = random.seed(random_seed)

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(self.state_size, self.action_size, random_seed).to(device)
        self.actor_target = Actor(self.state_size, self.action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=params['lr_actor'])

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(self.state_size, self.action_size, random_seed).to(device)
        self.critic_target = Critic(self.state_size, self.action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=params['lr_critic'],
                                           weight_decay=params['weight_decay'])

        # Noise process
        self.noise = OUNoise(self.action_size, random_seed)



    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, self.tau)
        self.soft_update(self.actor_local, self.actor_target, self.tau)

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)




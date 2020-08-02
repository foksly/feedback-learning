import typing as tp
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from agent import StatePreprocessor
from utils import ignore_keyboard_traceback

import matplotlib.pyplot as plt
from IPython import display

class A2CAgent(nn.Module):
    def __init__(self,
                 coord_range,
                 field_values_range,
                 n_completed_range,
                 coord_emb_dim,
                 field_emb_dim,
                 completed_emb_dim,
                 field_shape,
                 receptive_field=3,
                 n_actions=3,
                 epsilon=0.1):
        super().__init__()
        self.n_actions = n_actions
        self.preprocessor = StatePreprocessor(coord_range, field_values_range,
                                              n_completed_range, coord_emb_dim,
                                              field_emb_dim, completed_emb_dim,
                                              field_shape[1], receptive_field=receptive_field)
        
        self.backbone = nn.Sequential(
            nn.Linear(self.preprocessor.state_dim, 512), nn.LeakyReLU(),
            nn.Linear(512, 256), nn.LeakyReLU(), 
            nn.Linear(256, 128), nn.LeakyReLU())
        self.policy_head = nn.Sequential(nn.Linear(128, 64), nn.ELU(),
                                         nn.Linear(64, n_actions))
        self.value_head = nn.Sequential(nn.Linear(128, 64), nn.ELU(),
                                         nn.Linear(64, 1))


    def forward(self, states):
        state_enc = self.preprocessor(*self.preprocessor.to_tensor(states))
        backbone = self.backbone(state_enc)
        logps = F.log_softmax(self.policy_head(backbone), -1)
        values = self.value_head(backbone)
        return logps, values
    

    def sample_actions(self, logps):
        actions = []
        probs = torch.exp(logps).detach().cpu().numpy()
        for p in probs:
            actions.append(np.random.choice(self.n_actions, p=p))
        return actions
    

    def get_action(self, state, greedy=False):
        with torch.no_grad():
            logps, _ = self.forward(state)
        probs = torch.exp(logps).squeeze().cpu().numpy()
        if greedy:
            return np.argmax(probs)
        return np.random.choice(self.n_actions, p=probs)


    def to_tensor(self, state):
        device = next(self.parameters()).device
        if isinstance(state, list):
            coord, obs, n_completed = state.coord, state.obs, state.n_completed
        else:
            coord, obs, n_completed = [state.coord], [state.obs], [state.n_completed]

        coord_t = torch.as_tensor(coord, device=device)
        obs_t = torch.as_tensor(obs, dtype=torch.long, device=device)
        n_completed_t = torch.as_tensor(n_completed, device=device)
        return coord_t, obs_t, n_completed_t


class Runner:
    """
    We use this class to generate batches of experiences
    __init__:
        - Initialize the runner
    get_batch():
        - Make a mini batch of experiences
    """
    def __init__(self, n_agents, agent, make_env, max_steps=100, use_hint=False):
        self.n_agents = n_agents
        self.agent = agent
        self.make_env = make_env
        self.env = make_env()
        self.envs = [make_env() for _ in range(n_agents)]
        self.max_steps = max_steps


    def _compute_logps(self):
        with torch.no_grad():
            if self.use_hint:
                hints = [
                    self.env.get_hint(state, hint_type=self.agent.hint_type)
                    for state in self.states
                ]
                logps, _ = self.agent(self.states, hints)
            else:
                logps, _ = self.agent(self.states)
        return logps


    def run(self, n_steps=5):
        self.agent.eval()
        mb_states, mb_actions, mb_rewards, mb_next_states, mb_dones = [], [], [], [], []

        for _ in range(n_steps):
            rewards, next_states, dones = [], [], []

            logps = self._compute_logps()
            actions = self.agent.sample_actions(logps)
            mb_states.append(self.states)
            for i, env in enumerate(self.envs):
                next_s, r, d = env.step(actions[i])
                rewards.append(r)
                next_states.append(next_s)
                dones.append(dones)
                if done or self.n_steps[i] == self.max_steps:
                    self.states[i] = env.reset()
                    self.n_steps[i] = 0
                else:
                    self.states[i] = next_s
                    self.n_steps[i] += 1

            mb_actions.append(actions)
            mb_rewards.append(rewards)
            mb_next_states.append(next_states)
            mb_dones.append(dones)
            
        return mb_states, mb_actions, mb_rewards, mb_next_states, mb_dones


    def get_trajectories(self):
        self.agent.eval()

        envs = [self.make_env() for _ in range(self.n_agents)]
        trajs = [[] for _ in range(self.n_agents)]
        states = [env.reset() for env in envs]
        dones = [False for _ in range(self.n_agents)]
        n_steps = 0

        while (n_steps < self.max_steps):
            n_steps += 1
            not_done_states = []
            for i, done in enumerate(dones):
                if not done:
                    not_done_states.append(states[i])
                
            if len(not_done_states) == 0:
                break
                
            with torch.no_grad():
                logps, _ = self.agent(not_done_states)
            
            actions = self.agent.sample_actions(logps)
            idx = 0
            for i, done in enumerate(dones):
                if not done:
                    next_s, r, done = envs[i].step(actions[idx])
                    trajs[i].append([states[i], actions[idx], r, next_s, done])
                    dones[i] = done
                    states[i] = next_s
                    idx += 1
                    

        return trajs


def compute_trajectory_loss(trajectory, agent, env, 
                            gamma=0.99, entropy_term_strength=0.02, 
                            device=torch.device('cuda')):
    states, actions, rewards, _, _ = map(list, zip(*trajectory))
    actions = torch.as_tensor(actions, device=device, dtype=torch.long)
    rewards = torch.as_tensor(rewards, device=device, dtype=torch.float)
    
    logps, values = agent(states)
    td_target = 0.
    np_values = values.view(-1).cpu().detach().numpy()
    td_targets = np.zeros(len(trajectory))
    advantages = np.zeros(len(trajectory))

    for i in range(len(trajectory) - 1, -1, -1):
        td_target = rewards[i] + gamma * td_target
        advantage = td_target - np_values[i]
        td_targets[i] = td_target
        advantages[i] = advantage

    chosen_action_log_probs = logps.gather(1, actions.view(-1, 1))
    
    advantages_tensor = torch.as_tensor(advantages, device=device)
    policy_loss = -torch.sum(chosen_action_log_probs.squeeze() * advantages_tensor)
    
    td_targets_tensor = torch.as_tensor(td_targets, device=device)
    value_loss = torch.sum(torch.pow(td_targets_tensor - values.squeeze(), 2))

    entropy_loss = -torch.sum(logps * torch.exp(logps))

    return policy_loss + 0.5 * value_loss - entropy_term_strength * entropy_loss


def compute_loss(trajectories, agent, env, 
                 gamma=0.99, entropy_term_strength=0.02, 
                 device=torch.device('cuda')):
    loss = None
    for trajectory in trajectories:
        if loss is None:
            loss = compute_trajectory_loss(trajectory, agent, env, gamma, 
                                           entropy_term_strength, device)
        else:
            loss += compute_trajectory_loss(trajectory, agent, env, gamma, 
                                            entropy_term_strength, device)
    return loss / len(trajectories)


def plot_progress(loss_log, reward_log):
    _, axs = plt.subplots(1, 2, figsize=(15, 5))

    axs[0].plot(loss_log)
    axs[0].set_title('TD loss history')

    axs[1].plot(reward_log)
    axs[1].set_title('Mean reward per episode')

    plt.show()

@ignore_keyboard_traceback
def train(n_epochs,
          make_env,
          agent,
          optimizer,
          runner,
          max_steps=50,
          device=torch.device('cuda'),
          reward_log_freq=10,
          plot_every=100):
    
    agent.to(device)
    env = make_env()
    loss_log = []
    reward_log = []
    
    for epoch in range(n_epochs):
        trajectories = runner.get_trajectories()
        
        agent.train()
        optimizer.zero_grad()
        loss = compute_loss(trajectories, agent, env, device=device)
        loss_log.append(loss.item())

        loss.backward()
        optimizer.step()
            
        if (epoch + 1) % reward_log_freq == 0:
            trajs_rewards = []
            for trajectory in trajectories:
                _, _, rewards, _, _ = map(list, zip(*trajectory))
                trajs_rewards.append(sum(rewards))
            reward_log.append(np.mean(trajs_rewards))
        
        if (epoch + 1) % plot_every == 0:
            display.clear_output()
            print(f'Epoch #{epoch + 1}')
            plot_progress(loss_log, reward_log)

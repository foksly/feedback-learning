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
        probs = torch.exp(logps).cpu().numpy()
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
        self.env = make_env()
        self.envs = [make_env() for _ in range(n_agents)]
        self.states = [env.reset() for env in self.envs]

        self.max_steps = max_steps
        self.n_steps = [0 for _ in range(n_agents)]
        self.use_hint = use_hint

    def run(self):
        self.agent.eval()
        mb_states, mb_actions, mb_rewards, mb_next_states, mb_dones = [], [], [], [], []

        with torch.no_grad():
            if self.use_hint:
                hints = [
                    self.env.get_hint(state, hint_type=self.agent.hint_type)
                    for state in self.states
                ]
                logps, _ = self.agent(self.states, hints)
            else:
                logps, _ = self.agent(self.states)

        mb_actions = self.agent.sample_actions(logps)
        for i, env in enumerate(self.envs):
            next_state, reward, done = env.step(mb_actions[i])

            mb_states.append(self.states[i])
            mb_rewards.append(reward)
            mb_next_states.append(next_state)
            mb_dones.append(done)

            if done or self.n_steps[i] == self.max_steps:
                self.states[i] = env.reset()
                self.n_steps[i] = 0
            else:
                self.states[i] = next_state
                self.n_steps[i] += 1
            
        return mb_states, mb_actions, mb_rewards, mb_next_states, mb_dones


def evaluate(agent, env, max_steps=100):
    agent.eval()

    s = env.reset()
    total_reward = 0
    n_steps = 0
    done = False
    while (not done and n_steps < max_steps):
        a = agent.get_action(s, greedy=True)
        next_s, r, done = env.step(a)
        total_reward += r
        s = next_s
        n_steps += 1
    return total_reward


def compute_loss(states,
                 actions,
                 rewards,
                 next_states,
                 dones,
                 agent,
                 env,
                 gamma=0.99,
                 use_hints=False,
                 entropy_term_strength=0.02,
                 device=torch.device('cuda')):
    actions = torch.as_tensor(actions, device=device, dtype=torch.long)
    rewards = torch.as_tensor(rewards, device=device, dtype=torch.float)

    is_done = torch.as_tensor(np.array(dones, dtype=np.float32),
                                       device=device, dtype=torch.float)
    is_not_done = 1 - is_done
    
    if use_hints:
        hints = [   
            env.get_hint(state, hint_type=agent.hint_type) for state in states
        ]
        next_hints = [
            env.get_hint(state, hint_type=agent.hint_type)
            for state in next_states
        ]

    logps, values = agent(states, hints) if use_hints else agent(states)
    _, next_values = agent(next_states,
                           next_hints) if use_hints else agent(states)

    td_target = rewards + gamma * (next_values.squeeze() * is_not_done)
    advantage = td_target - values

    chosen_action_log_probs = logps.gather(1, actions.view(-1, 1))
    policy_loss = -torch.sum(chosen_action_log_probs * advantage.detach())

    value_loss = torch.sum(torch.pow(td_target.detach() - values, 2)) # ?

    entropy_loss = torch.sum(-logps * torch.exp(logps))

    return policy_loss + 0.5 * value_loss - entropy_term_strength * entropy_loss

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
        states, actions, rewards, next_states, dones = runner.run()
        
        agent.train()
        optimizer.zero_grad()
        loss = compute_loss(states, actions, rewards, next_states, dones, agent, env)
        loss_log.append(loss.item())

        loss.backward()
        optimizer.step()
            
        if (epoch + 1) % reward_log_freq == 0:
            reward_log.append(evaluate(agent, make_env(), max_steps))
        
        if (epoch + 1) % plot_every == 0:
            display.clear_output()
            print(f'Epoch #{epoch + 1}')
            plot_progress(loss_log, reward_log)


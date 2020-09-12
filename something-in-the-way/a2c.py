import typing as tp

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from embedders import StateEncoder, HintEncoder
from utils import ignore_keyboard_traceback, save_model

import matplotlib.pyplot as plt
from IPython import display


class A2CAgent(nn.Module):
    def __init__(self,
                 state_config,
                 hint_type=None,
                 hint_config=None,
                 receptive_field=3,
                 n_actions=4,
                 epsilon=0.1):
        super().__init__()
        self.n_actions = n_actions
        self.hint_type = hint_type

        self.state_encoder = StateEncoder(state_config,
                                          receptive_field=receptive_field)
        hint_dim = 0
        if hint_type is not None:
            self.hint_encoder = HintEncoder(hint_type,
                                            hint_config)  # code hint type
            hint_dim = self.hint_encoder.hint_dim

        self.backbone = nn.Sequential(
            nn.Linear(self.state_encoder.state_dim + hint_dim, 512),
            nn.LeakyReLU(), nn.Linear(512, 256), nn.LeakyReLU(),
            nn.Linear(256, 128), nn.LeakyReLU())
        self.policy_head = nn.Sequential(nn.Linear(128, 64), nn.ELU(),
                                         nn.Linear(64, n_actions))
        self.value_head = nn.Sequential(nn.Linear(128, 64), nn.ELU(),
                                        nn.Linear(64, 1))

    def forward(self, states, hint=None):
        backbone_input = self.state_encoder(states)
        if hint is not None:
            hint_enc = self.hint_encoder(hint)
            backbone_input = torch.cat([backbone_input, hint_enc], dim=1)  #

        backbone = self.backbone(backbone_input)
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
            coord, obs, n_completed = [state.coord], [state.obs
                                                      ], [state.n_completed]

        coord_t = torch.as_tensor(coord, device=device)
        obs_t = torch.as_tensor(obs, dtype=torch.long, device=device)
        n_completed_t = torch.as_tensor(n_completed, device=device)
        return coord_t, obs_t, n_completed_t


def get_trajectories(n_agents, agent, make_env, max_steps=100, hint_type=None):
    agent.eval()

    env = make_env()
    envs = [make_env() for _ in range(n_agents)]
    trajs = [[] for _ in range(n_agents)]
    states = [env.reset() for env in envs]
    dones = [False for _ in range(n_agents)]
    n_steps = [0 for _ in range(n_agents)]

    while True:
        not_done_states = []
        not_done_hints = [] if hint_type is not None else None
        for i, done in enumerate(dones):
            if not done:
                not_done_states.append(states[i])
                if hint_type is not None:
                    not_done_hints.append(env.get_hint(states[i], hint_type))

        if len(not_done_states) == 0:
            break

        with torch.no_grad():
            logps, _ = agent(not_done_states, not_done_hints)

        actions = agent.sample_actions(logps)
        idx = 0
        for i, done in enumerate(dones):
            if not done:
                next_s, r, done = envs[i].step(actions[idx])
                trajs[i].append([states[i], actions[idx], r, next_s, done])
                if hint_type is not None:
                    hint = env.get_hint(states[i], hint_type=hint_type)
                    trajs[i][-1].append(hint)

                states[i] = next_s
                dones[i] = done
                if r > 0:
                    n_steps[i] = 0
                else:
                    n_steps[i] += 1
                if n_steps[i] >= max_steps:
                    dones[i] = True

                idx += 1

    return trajs


def compute_trajectory_loss(trajectory,
                            agent,
                            env,
                            hint_type=None,
                            gamma=0.99,
                            entropy_term_strength=0.02,
                            device=torch.device('cuda')):
    data = list(map(list, zip(*trajectory)))
    # states, actions, rewards, _, _ = map(list, zip(*trajectory))
    states, actions, rewards = data[:3]
    hints = data[-1] if hint_type is not None else None

    actions = torch.as_tensor(actions, device=device, dtype=torch.long)
    rewards = torch.as_tensor(rewards, device=device, dtype=torch.float)

    logps, values = agent(states, hints)
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
    policy_loss = -torch.sum(
        chosen_action_log_probs.squeeze() * advantages_tensor)

    td_targets_tensor = torch.as_tensor(td_targets, device=device)
    value_loss = torch.sum(torch.pow(td_targets_tensor - values.squeeze(), 2))

    entropy_loss = -torch.sum(logps * torch.exp(logps))

    return policy_loss + 0.5 * value_loss - entropy_term_strength * entropy_loss


def compute_loss(trajectories,
                 agent,
                 env,
                 hint_type=None,
                 gamma=0.99,
                 entropy_term_strength=0.02,
                 device=torch.device('cuda')):
    loss = None
    for trajectory in trajectories:
        if loss is None:
            loss = compute_trajectory_loss(trajectory, agent, env, hint_type,
                                           gamma, entropy_term_strength,
                                           device)
        else:
            loss += compute_trajectory_loss(trajectory, agent, env, hint_type,
                                            gamma, entropy_term_strength,
                                            device)
    return loss / len(trajectories)


def plot_progress(loss_log, reward_log, valid_reward_log=None):
    _, axs = plt.subplots(1, 2, figsize=(15, 5))

    axs[0].plot(loss_log)
    axs[0].set_title('TD loss history')

    axs[1].plot(reward_log, label='Train')
    axs[1].set_title('Mean reward per episode')
    if valid_reward_log is not None:
        axs[1].plot(valid_reward_log, label='Valid')

    plt.legend()
    plt.show()


def compute_mean_reward(trajectories):
    trajs_rewards = []
    for trajectory in trajectories:
        rewards = list(map(list, zip(*trajectory)))[2]
        trajs_rewards.append(sum(rewards))
    return np.mean(trajs_rewards)


@ignore_keyboard_traceback
def train(n_epochs,
          n_agents,
          train_factory,
          agent,
          optimizer,
          max_steps=50,
          hint_type=None,
          test_factory=None,
          device=torch.device('cuda'),
          experiment_name=None,
          save_path=None,
          log_dir=None,
          checkpoint_every=100,
          max_reward_limit=10,
          reward_log_freq=10,
          print_progress=True,
          plot_progress=False,
          plot_every=100):
    agent.to(device)
    loss_log = []
    reward_log = []
    max_reward_hits = 0
    max_reward = train_factory.get_max_reward()
    if test_factory is not None:
        test_reward_log = []

    for epoch in range(1, n_epochs):
        train_factory.switch()
        trajectories = get_trajectories(n_agents,
                                        agent,
                                        train_factory,
                                        max_steps=max_steps,
                                        hint_type=hint_type)

        agent.train()
        optimizer.zero_grad()
        loss = compute_loss(trajectories,
                            agent,
                            train_factory(),
                            hint_type=hint_type,
                            device=device)
        loss_log.append(loss.item())

        loss.backward()
        optimizer.step()

        if save_path is not None and epoch % checkpoint_every == 0:
            save_model(save_path, agent.state_dict(), epoch)

        if epoch % reward_log_freq == 0:
            mean_reward = compute_mean_reward(trajectories)
            reward_log.append(mean_reward)

            if test_factory is not None:
                test_mean_rewards = []
                for i in range(len(test_factory.configs)):
                    test_factory.id = i
                    test_trajectories = get_trajectories(n_agents,
                                                         agent,
                                                         test_factory,
                                                         max_steps=max_steps,
                                                         hint_type=hint_type)
                    mean_reward = compute_mean_reward(test_trajectories)
                    test_mean_rewards.append(mean_reward)
                total_mean_reward = np.mean(test_mean_rewards)
                test_reward_log.append(total_mean_reward)

            if print_progress:
                print(f'\nEpoch #{epoch}')
                print(f'Mean reward...{mean_reward} \n')
                print()

            if mean_reward > max_reward:
                max_reward_hits += 1

            if max_reward_hits == max_reward_limit:
                break

        if plot_progress and epoch % plot_every == 0:
            display.clear_output()
            print(f'Epoch #{epoch + 1}')
            plot_progress(loss_log, reward_log)

    if test_factory is None:
        return reward_log

    return reward_log, test_reward_log


@ignore_keyboard_traceback
def train_meta(n_epochs,
               n_agents,
               envs_factories,
               valid_env_factories,
               agent,
               optimizer,
               max_steps=50,
               device=torch.device('cuda'),
               reward_log_freq=10,
               valid_reward_log_freq=20,
               plot_every=100):
    agent.to(device)
    loss_log = []
    reward_log = []
    valid_reward_log = []

    for epoch in range(n_epochs):
        make_env = envs_factories[np.random.randint(len(envs_factories))]
        env = make_env()

        trajectories = get_trajectories(n_agents,
                                        agent,
                                        make_env,
                                        max_steps=max_steps)
        agent.train()
        optimizer.zero_grad()
        loss = compute_loss(trajectories, agent, env, device=device)
        loss_log.append(loss.item())

        loss.backward()
        optimizer.step()

        if (epoch + 1) % reward_log_freq == 0:
            epoch_env_mean_reward = compute_mean_reward(trajectories)
            reward_log.append(epoch_env_mean_reward)

        if (epoch + 1) % valid_reward_log_freq == 0:
            valid_mean_rewards = []
            for valid_make_env in valid_env_factories:
                valid_trajs = get_trajectories(n_agents,
                                               agent,
                                               valid_make_env,
                                               max_steps=max_steps)
                valid_mean_rewards.append(compute_mean_reward(valid_trajs))
            valid_reward_log.append(np.mean(valid_mean_rewards))

        if (epoch + 1) % plot_every == 0:
            display.clear_output()
            print(f'Epoch #{epoch + 1}')
            plot_progress(loss_log, reward_log, valid_reward_log)
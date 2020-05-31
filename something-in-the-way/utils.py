import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from functools import wraps
from IPython import display

import torch
import torch.nn.functional as F
from torch import nn

# ---------------------------------------------
#               Replay buffer
# ---------------------------------------------

class ReplayBuffer(object):
    def __init__(self, size):
        """Create Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def add(self, trajectory):
        if self._next_idx >= len(self._storage):
            self._storage.append(trajectory)
        else:
            self._storage[self._next_idx] = trajectory
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def sample(self, batch_size):
        """Sample a batch of experiences.
        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        Returns
        -------
        coord_batch: np.array
            batch of coordinates
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        coords_next_batch: np.array
            next set of observations seen after executing act_batch
        obses_next_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        coords, obses, actions, rewards, coords_next, obses_next, dones = [], [], [], [], [], [], []
        total_transitions = sum(len(t) for t in self._storage)
        probs = [len(t) / total_transitions for t in self._storage]
        traj_idxs = np.random.choice(len(self._storage), batch_size, p=probs)
        for i in traj_idxs:
            idx = np.random.choice(len(self._storage[i]))
            data = self._storage[i][idx]
            state, action, reward, state_next, done = data
            coords.append(np.array(state[0], copy=False))
            obses.append(state[1])
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            coords_next.append(np.array(state_next[0], copy=False))
            obses_next.append(state_next[1])
            dones.append(done)
        return (np.array(coords), np.array(obses), np.array(actions),
                np.array(rewards), np.array(coords_next), np.array(obses_next),
                np.array(dones))


def collect_trajectories(n_trajs, agent, env, exp_replay, max_steps=20):
    for _ in range(n_trajs):
        trajectory = []
        s = env.reset()
        n_steps = 0
        done = False

        total_reward = 0
        while (not done and n_steps < max_steps):
            qvalues = agent.get_qvalues(s)
            a = agent.sample_actions(qvalues).item()
            next_s, r, done = env.step(a)
            total_reward += r
            trajectory.append([s, a, r, next_s, done])
            s = next_s
            n_steps += 1
        exp_replay.add(trajectory)


def evaluate(n_trajs, agent, env, max_steps=20):
    rewards = []
    for _ in range(n_trajs):
        s = env.reset()
        total_reward = 0
        n_steps = 0
        done = False
        while (not done and n_steps < max_steps):
            qvalues = agent.get_qvalues(s)
            a = agent.sample_actions(qvalues).item()
            next_s, r, done = env.step(a)
            total_reward += r
            s = next_s
            n_steps += 1
        rewards.append(total_reward)
    return np.mean(rewards)

# ---------------------------------------------
#               Train funcs
# ---------------------------------------------

def compute_td_loss(coords,
                    obses,
                    actions,
                    rewards,
                    next_coords,
                    next_obses,
                    is_done,
                    agent,
                    target_network,
                    gamma=0.99,
                    device=torch.device('cuda')):
    """ 
    Compute td loss using torch operations only. Use the formulae above. 
    Parameters
        ----------
        coords: np.array (batch_size, 2)
        obses: np.array (batch_size, receptive_field, receptive_field)
    """
    coords = torch.as_tensor(coords, device=device)  # shape: [batch_size, 2]
    obses = torch.as_tensor(
        obses, dtype=torch.long,
        device=device)  #shape [batch_size, receptive_field, receptive_field]

    # for some torch reason should not make actions a tensor
    actions = torch.as_tensor(actions, device=device,
                              dtype=torch.long)  # shape: [batch_size]
    rewards = torch.as_tensor(rewards, device=device,
                              dtype=torch.float)  # shape: [batch_size]
    # shape: [batch_size, *state_shape]
    next_coords = torch.as_tensor(next_coords,
                                  device=device)  # shape: [batch_size, 2]
    next_obses = torch.as_tensor(next_obses, dtype=torch.long, device=device)
    is_done = torch.as_tensor(is_done.astype('float32'),
                              device=device,
                              dtype=torch.float)  # shape: [batch_size]
    is_not_done = 1 - is_done

    # get q-values for all actions in current states
    predicted_qvalues = agent(coords, obses)

    # compute q-values for all actions in next states
    predicted_next_qvalues_target = target_network(next_coords, next_obses)
    predicted_next_qvalues_agent = agent(next_coords, next_obses)

    # select q-values for chosen actions
    predicted_qvalues_for_actions = predicted_qvalues[range(len(actions)),
                                                      actions]

    # compute V*(next_states) using predicted next q-values
    next_actions = predicted_next_qvalues_agent.argmax(dim=1)
    next_state_values = predicted_next_qvalues_target[range(len(next_actions)),
                                                      next_actions]

    target_qvalues_for_actions = rewards + gamma * (next_state_values *
                                                    is_not_done)

    # MSE
    loss = torch.mean((predicted_qvalues_for_actions -
                       target_qvalues_for_actions.detach())**2)

    return loss


def plot_progress(loss_log, reward_log):
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    axs[0].plot(loss_log)
    axs[0].set_title('TD loss history')

    axs[1].plot(reward_log)
    axs[1].set_title('Mean reward per episode')

    plt.show()


def ignore_keyboard_traceback(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except KeyboardInterrupt:
            pass

    return wrapper


@ignore_keyboard_traceback
def train(n_epochs,
          env,
          agent,
          target_network,
          optimizer,
          exp_replay,
          epsilon,
          trajs_per_epoch,
          batch_size,
          device=torch.device('cuda'),
          refresh_target_network_freq=100,
          loss_log_freq=20,
          plot_every=100):

    agent.to(device)
    target_network.to(device)

    loss_log = []
    reward_log = []

    env.reset()
    for epoch in range(n_epochs + 1):
        agent.epsilon = epsilon.get_value(epoch)
        collect_trajectories(trajs_per_epoch, agent, env, exp_replay)

        optimizer.zero_grad()

        # loss
        coords, obs, a, r, next_coords, next_obs, is_done = exp_replay.sample(
            batch_size)
        loss = compute_td_loss(coords,
                               obs,
                               a,
                               r,
                               next_coords,
                               next_obs,
                               is_done,
                               agent,
                               target_network,
                               device=device)
        loss.backward()
        optimizer.step()

        if epoch % loss_log_freq == 0:
            loss_log.append(loss.item())

        if epoch % plot_every == 0:
            reward_log.append(evaluate(100, agent, env))
            display.clear_output()
            print(f'Epoch #{epoch}')
            plot_progress(loss_log, reward_log)

        if epoch % refresh_target_network_freq == 0:
            target_network.load_state_dict(agent.state_dict())


# ---------------------------------------------
#               Epsilon
# ---------------------------------------------


class Epsilon(ABC):
    @abstractmethod
    def get_value(self, episode):
        pass


class LinearEpsilon(Epsilon):
    def __init__(self, total_steps, min_value=0.01, max_value=0.9):
        self.total_steps = total_steps
        self.min_value = min_value
        self.max_value = max_value

    def get_value(self, episode):
        if episode >= self.total_steps:
            return self.min_value
        return (self.max_value * (self.total_steps - episode) +
                self.min_value * episode) / self.total_steps


class ExpEpsilon(Epsilon):
    def __init__(self, min_value=0.01, max_value=0.9, gamma=1e-3):
        self.min_epsilon = 0.01
        self.max_epsilon = 0.9
        self.gamma = gamma

    def get_value(self, episode):
        return self.min_epsilon + (self.max_epsilon - self.min_epsilon
                                   ) * np.exp(-self.gamma * episode)

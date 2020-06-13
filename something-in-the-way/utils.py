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
        coords, obses, n_completed, actions, rewards = [], [], [], [], []
        coords_next, obses_next, n_completed_next, dones = [], [], [], []
        total_transitions = sum(len(t) for t in self._storage)
        probs = [len(t) / total_transitions for t in self._storage]
        traj_idxs = np.random.choice(len(self._storage), batch_size, p=probs)
        for i in traj_idxs:
            idx = np.random.choice(len(self._storage[i]))
            data = self._storage[i][idx]
            state, action, reward, state_next, done = data
            coords.append(np.array(state[0], copy=False))
            obses.append(state[1])
            n_completed.append(np.array([state[2]]))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            coords_next.append(np.array(state_next[0], copy=False))
            obses_next.append(state_next[1])
            n_completed_next.append(np.array([state[2]]))
            dones.append(done)
        return (np.array(coords), np.array(obses), np.array(n_completed), 
                np.array(actions), np.array(rewards), np.array(coords_next), 
                np.array(obses_next), np.array(n_completed_next), np.array(dones))
    
    def sample_window(self, batch_size, window_size=-1):
        coords, obses, n_completed, actions, rewards = [], [], [], [], []
        coords_next, obses_next, n_completed_next, dones = [], [], [], []
        total_transitions = sum(len(t) for t in self._storage)
        probs = [len(t) / total_transitions for t in self._storage]
        traj_idxs = np.random.choice(len(self._storage), batch_size, p=probs)
        for i in traj_idxs:
            idx = np.random.choice(len(self._storage[i]))
            from_ = 0 if window_size == -1 else max(0, idx - window_size)
            data = self._storage[i][from_:idx + 1]
            i_coords, i_obses, i_n_completed = [], [], []
            i_coords_next, i_obses_next, i_n_completed_next = [], [], []
            for d in data:
                state, action, reward, state_next, _ = d
                i_coords.append(np.array(state[0], copy=False))
                i_obses.append(state[1])
                i_n_completed.append(np.array([state[2]]))
                i_coords_next.append(np.array(state_next[0], copy=False))
                i_obses_next.append(state_next[1])
                i_n_completed_next.append(np.array([state[2]]))
            
            coords.append(i_coords)
            obses.append(i_obses_next)
            n_completed.append(i_n_completed)
            coords_next.append(i_coords_next)
            obses_next.append(i_obses_next)
            n_completed_next.append(i_n_completed)

            state, action, reward, state_next, done = data[-1]
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            dones.append(done)
        return (coords, obses, n_completed, 
                np.array(actions), np.array(rewards), coords_next, 
                obses_next, n_completed_next, np.array(dones))



def collect_trajectories(n_trajs, agent, env, exp_replay, max_steps=20, use_hints=True):
    for _ in range(n_trajs):
        trajectory = []
        s = env.reset()
        n_steps = 0
        done = False

        total_reward = 0
        if use_hints:
            hints = env.get_hints()
        while (not done and n_steps < max_steps):
            if use_hints:
                qvalues = agent.get_qvalues(s, hints)
            else:
                qvalues = agent.get_qvalues(s)
            a = agent.sample_actions(qvalues).item()
            next_s, r, done = env.step(a)
            total_reward += r
            trajectory.append([s, a, r, next_s, done])
            s = next_s
            n_steps += 1
        exp_replay.add(trajectory)


def collect_trajectories_batch(n_trajs, agent, make_env, 
                               exp_replay, max_steps=20, 
                               use_hints=True, device=torch.device('cuda')):

    agent.eval()
    envs = [make_env() for _ in range(n_trajs)]
    trajs = [[] for _ in range(n_trajs)]
    states = [env.reset() for env in envs]
    n_steps = 0
    dones = [False for _ in range(n_trajs)]
    if use_hints:
        hints_coords, hints_obses, hints_actions = hints2tensor(envs[0].get_hints())

    while (n_steps < max_steps):
        n_steps += 1
        coords = []
        obses = []
        n_completed = []

        for i in range(len(dones)):
            if not dones[i]:
                coords.append(np.array(states[i][0], copy=False))
                obses.append(states[i][1])
                n_completed.append(np.array([states[i][2]]))
        if len(coords) == 0:
            break
            
        coords_t = torch.as_tensor(np.array(coords), device=device)
        obses_t = torch.as_tensor(np.array(obses), dtype=torch.long, device=device) 
        n_completed_t = torch.as_tensor(np.array(n_completed), device=device)

        with torch.no_grad():
            if use_hints:
                qvalues = agent(coords_t, obses_t, n_completed_t,
                                hints_coords, hints_obses, hints_actions)
            else:
                qvalues = agent(coords_t, obses_t, n_completed_t)
        
        actions = agent.sample_actions(qvalues.cpu().numpy())
        idx = 0
        for i in range(len(dones)):
            if not dones[i]:
                next_s, r, done = envs[i].step(actions[idx])
                trajs[i].append([states[i], actions[idx], r, next_s, done])
                if done:
                    dones[i] = True
                states[i] = next_s
                idx += 1
    for traj in trajs:
        exp_replay.add(traj)


def evaluate(n_trajs, agent, env, max_steps=20, use_hints=True):
    rewards = []
    for _ in range(n_trajs):
        s = env.reset()
        total_reward = 0
        n_steps = 0
        done = False
        if use_hints:
            hints = env.get_hints()
        while (not done and n_steps < max_steps):
            if use_hints:
                qvalues = agent.get_qvalues(s, hints)
            else:
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

def hints2tensor(hints, device=torch.device('cuda')):
    coords, obs, action = [], [], []
    for hint in hints:
        coords.append(np.array(hint[0].coord, copy=False))
        obs.append(hint[0].obs)
        action.append(np.array([hint[1]]))
    coords_t = torch.as_tensor(coords, device=device)
    obs_t = torch.as_tensor(obs, dtype=torch.long, device=device)
    action_t = torch.as_tensor(action, device=device)
    return coords_t, obs_t, action_t

def compute_td_loss(coords,
                    obses,
                    n_completed,
                    actions,
                    rewards,
                    next_coords,
                    next_obses,
                    next_n_completed,
                    is_done,
                    agent,
                    target_network,
                    hints=None,
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
    n_completed = torch.as_tensor(n_completed, device=device)

    # for some torch reason should not make actions a tensor
    actions = torch.as_tensor(actions, device=device,
                              dtype=torch.long)  # shape: [batch_size]
    rewards = torch.as_tensor(rewards, device=device,
                              dtype=torch.float)  # shape: [batch_size]
    # shape: [batch_size, *state_shape]
    next_coords = torch.as_tensor(next_coords,
                                  device=device)  # shape: [batch_size, 2]
    next_obses = torch.as_tensor(next_obses, dtype=torch.long, device=device)
    next_n_completed = torch.as_tensor(next_n_completed, device=device)
    is_done = torch.as_tensor(is_done.astype('float32'),
                              device=device,
                              dtype=torch.float)  # shape: [batch_size]
    
    if hints is not None:
        hints_coords, hints_obses, hints_actions = hints2tensor(hints)
    is_not_done = 1 - is_done

    if hints is not None:
        predicted_qvalues = agent(coords, obses, n_completed, 
                                  hints_coords, hints_obses, hints_actions)
        predicted_next_qvalues_target = target_network(next_coords, next_obses, next_n_completed,
                                                    hints_coords, hints_obses, hints_actions)
        predicted_next_qvalues_agent = agent(next_coords, next_obses, next_n_completed,
                                            hints_coords, hints_obses, hints_actions)
    else:
        predicted_qvalues = agent(coords, obses, n_completed)
        predicted_next_qvalues_target = target_network(next_coords, next_obses, next_n_completed)
        predicted_next_qvalues_agent = agent(next_coords, next_obses, next_n_completed)

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


def save_model(path: str, model_state_dict,
               epoch: int, reward: int) -> None:
    """
    Saves model
    :param path: path where to save model
    """
    torch.save(
        {
            'model_state_dict': model_state_dict,
            'epoch': epoch,
            'reward': reward
        }, path)


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
          hints,
          epsilon,
          trajs_per_epoch,
          batch_size,
          max_steps=50,
          device=torch.device('cuda'),
          refresh_target_network_freq=100,
          loss_log_freq=20,
          reward_log_freq=10,
          plot_every=100,
          loss_clip=2,
          min_reward_for_saving=5,
          save_path='agent.pt'):

    agent.to(device)
    target_network.to(device)

    loss_log = []
    reward_log = []
    best_seen_reward = min_reward_for_saving

    use_hints = False
    if hints is not None:
        use_hints = True

    env.reset()
    for epoch in range(n_epochs + 1):
        agent.epsilon = epsilon.get_value(epoch)
        
        collect_trajectories(trajs_per_epoch, agent, env, exp_replay, max_steps, use_hints=use_hints)

        optimizer.zero_grad()

        # loss
        coords, obs, n_completed, a, r, \
        next_coords, next_obs, next_n_completed, is_done = exp_replay.sample(batch_size)
        loss = compute_td_loss(coords,
                               obs,
                               n_completed,
                               a,
                               r,
                               next_coords,
                               next_obs,
                               next_n_completed,
                               is_done,
                               agent,
                               target_network,
                               hints,
                               device=device)
        loss.backward()
        optimizer.step()

        if epoch % loss_log_freq == 0:
            if loss.item() > loss_clip:
                loss_log.append(loss_clip)
            else:
                loss_log.append(loss.item())
        
        if epoch % reward_log_freq == 0:
            agent.epsilon = 0.1
            reward_log.append(evaluate(10, agent, env, max_steps, use_hints=use_hints))
            if best_seen_reward < reward_log[-1]:
                best_seen_reward = reward_log[-1]
                save_model(save_path, agent.state_dict(), epoch, best_seen_reward)

        if epoch % plot_every == 0:
            display.clear_output()
            print(f'Epoch #{epoch}')
            plot_progress(loss_log, reward_log)

        if epoch % refresh_target_network_freq == 0:
            target_network.load_state_dict(agent.state_dict())


@ignore_keyboard_traceback
def train_async(n_epochs,
                make_env,
                agent,
                target_network,
                optimizer,
                exp_replay,
                hints,
                epsilon,
                trajs_per_epoch,
                batches_per_epoch,
                batch_size,
                max_steps=50,
                device=torch.device('cuda'),
                refresh_target_network_freq=100,
                loss_log_freq=20,
                reward_log_freq=10,
                plot_every=100,
                loss_clip=2,
                min_reward_for_saving=5,
                save_path='agent.pt'):

    agent.to(device)
    target_network.to(device)

    loss_log = []
    reward_log = []
    best_seen_reward = min_reward_for_saving

    use_hints = False
    if hints is not None:
        use_hints = True
    
    for epoch in range(n_epochs + 1):
        agent.epsilon = epsilon.get_value(epoch)
        
        collect_trajectories_batch(trajs_per_epoch, agent, make_env, exp_replay, max_steps, use_hints=use_hints)

        optimizer.zero_grad()

        # loss
        agent.train()
        batch_loss = 0
        for _ in range(batches_per_epoch):
            coords, obs, n_completed, a, r, \
            next_coords, next_obs, next_n_completed, is_done = exp_replay.sample(batch_size)
            loss = compute_td_loss(coords,
                                   obs,
                                   n_completed,
                                   a,
                                   r,
                                   next_coords,
                                   next_obs,
                                   next_n_completed,
                                   is_done,
                                   agent,
                                   target_network,
                                   hints,
                                   device=device)
            batch_loss += loss.item()
            loss.backward()
            optimizer.step()

        if epoch % loss_log_freq == 0:
            batch_loss /= batches_per_epoch
            if batch_loss > loss_clip:
                loss_log.append(loss_clip)
            else:
                loss_log.append(batch_loss)
        
        if epoch % reward_log_freq == 0:
            agent.epsilon = 0.1
            reward_log.append(evaluate(5, agent, make_env(), max_steps, use_hints=use_hints))
            if best_seen_reward < reward_log[-1]:
                best_seen_reward = reward_log[-1]
                save_model(save_path, agent.state_dict(), epoch, best_seen_reward)

        if epoch % plot_every == 0:
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

import heapq
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from functools import wraps
from itertools import count
from IPython import display

import torch
import torch.nn.functional as F
from torch import nn

# ---------------------------------------------
#               Replay buffer
# ---------------------------------------------

def add_transition(transition, coords, obses, n_completed, actions, rewards,
                   coords_next, obses_next, n_completed_next, dones, hints, next_hints):
    if hints is not None:
        state, action, reward, state_next, done, hint, next_hint = transition
    else:
        state, action, reward, state_next, done = transition

    coords.append(np.array(state[0]))
    obses.append(state[1])
    n_completed.append(np.array([state[2]]))
    actions.append(np.array(action))
    rewards.append(reward)
    coords_next.append(np.array(state_next[0]))
    obses_next.append(state_next[1])
    n_completed_next.append(np.array([state_next[2]]))
    dones.append(done)
    if hints is not None:
        hints.append(hint)
        next_hints.append(next_hint)

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
    
    def _get_transition(self, traj_idx, transition_idx):
        return self._storage[traj_idx][transition_idx]

    def _sample_transition_idx(self, traj_idx):
        return np.random.choice(len(self._storage[traj_idx]))

    def __len__(self):
        return len(self._storage)

    def add(self, trajectory):
        if self._next_idx >= len(self._storage):
            self._storage.append(trajectory)
        else:
            self._storage[self._next_idx] = trajectory
        self._next_idx = (self._next_idx + 1) % self._maxsize


    def sample(self, batch_size, add_hint=False):
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
        hints, next_hints = ([], []) if add_hint else (None, None)

        total_transitions = sum(len(t) for t in self._storage)
        probs = [len(t) / total_transitions for t in self._storage]
        traj_idxs = np.random.choice(len(self._storage), batch_size, p=probs)
        for i in traj_idxs:
            idx = self._sample_transition_idx(i)
            data = self._get_transition(i, idx)
            add_transition(data, coords, obses, n_completed, actions, rewards, 
                           coords_next, obses_next, n_completed_next, dones, hints, next_hints)

        batch = [np.array(coords), np.array(obses), np.array(n_completed), 
                np.array(actions), np.array(rewards), np.array(coords_next), 
                np.array(obses_next), np.array(n_completed_next), np.array(dones)]
        
        if hints is not None:
            batch.append(hints)
            batch.append(next_hints)

        return batch


class PriorityReplayBuffer(ReplayBuffer):
    def __init__(self, size):
        super().__init__(size)
        self._id = count()
    
    def add(self, trajectory, reward):
        if len(self._storage) < self._maxsize:
            heapq.heappush(self._storage, (reward, next(self._id), trajectory))
        if self._storage[0][0] < reward:
            heapq.heappop(self._storage)
            heapq.heappush(self._storage, (reward, next(self._id), trajectory))
    
    def _get_transition(self, traj_idx, transition_idx):
        return self._storage[traj_idx][2][transition_idx]
    
    def _sample_transition_idx(self, traj_idx):
        return np.random.choice(len(self._storage[traj_idx][2]))


def collect_trajectories(n_trajs, agent, env, exp_replay, 
                         max_steps=20, use_hints=True, priority=False):
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
        if priority:
            exp_replay.add(trajectory, total_reward)
        else:
            exp_replay.add(trajectory)


def collect_trajectories_batch(n_trajs, agent, make_env, 
                               exp_replay, priority_exp_replay=None,
                               max_steps=20, use_hints=True, 
                               device=torch.device('cuda')):

    agent.eval()
    envs = [make_env() for _ in range(n_trajs)]
    trajs = [[] for _ in range(n_trajs)]
    states = [env.reset() for env in envs]
    n_steps = 0
    dones = [False for _ in range(n_trajs)]
    rewards = [0 for _ in range(n_trajs)]
    if use_hints:
        hints_coords, hints_obses, hints_actions = hints2tensor(envs[0].get_hints())

    while (n_steps < max_steps):
        n_steps += 1
        coords = []
        obses = []
        n_completed = []

        for i in range(len(dones)):
            if not dones[i]:
                coords.append(np.array(states[i][0]))
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
                rewards[i] += r
                trajs[i].append([states[i], actions[idx], r, next_s, done])
                if done:
                    dones[i] = True
                states[i] = next_s
                idx += 1
    for i in range(len(trajs)):
        exp_replay.add(trajs[i])
        if priority_exp_replay is not None:
            priority_exp_replay.add(trajs[i], rewards[i])
            


def evaluate(n_trajs, agent, env, max_steps=20, use_hints=True):
    agent.eval()
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
        coords.append(np.array(hint.coord))
        obs.append(hint.obs)
        action.append(np.array([hint.n_completed]))
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


def plot_transitions(axs, transitions, field):
    init_color = np.array([0, 0, 90])
    addition = np.array([0, 0, 1])
    for transition in transitions:
        if np.sum(field[transition[0]][transition[1]]) == 0:
            field[transition[0]][transition[1]] = init_color
        elif field[transition[0]][transition[1]][2] < 250:
            field[transition[0]][transition[1]] += addition
    axs.imshow(field.astype(np.int))


def plot_progress(loss_log, reward_log, initial_state_v, grad_norm, 
                  transitions=None, priority_transitons=None, field=None):
    if transitions is not None:
        assert priority_transitons is not None and field is not None

        fig = plt.figure(constrained_layout=True)
        gs = fig.add_gridspec(4, 2)
    else:
        fig = plt.figure(constrained_layout=True)
        gs = fig.add_gridspec(2, 2)

    td_loss_plot = fig.add_subplot(gs[0, 0])
    td_loss_plot.plot(loss_log)
    td_loss_plot.set_title('TD loss history', fontsize=18)

    reward_plot = fig.add_subplot(gs[0, 1])
    reward_plot.plot(reward_log)
    reward_plot.set_title('Mean reward per episode', fontsize=18)

    init_state_v_plot = fig.add_subplot(gs[1, 0])
    init_state_v_plot.plot(initial_state_v)
    init_state_v_plot.set_title('Initial state V', fontsize=18)

    grad_norm_plot = fig.add_subplot(gs[1, 1])
    grad_norm_plot.plot(grad_norm)
    grad_norm_plot.set_title('Grad norm', fontsize=18)

    if transitions is not None:
        transition_plot = fig.add_subplot(gs[2, :])
        plot_transitions(transition_plot, transitions, field)
        transition_plot.set_title('Transitions sampled form experience replay', fontsize=18)

        priority_transitons_plot = fig.add_subplot(gs[3, :])
        plot_transitions(priority_transitons_plot, priority_transitons, field)
        priority_transitons_plot.set_title('Transitions sampled from prioritized experience replay', fontsize=18)


    # plt.tight_layout()
    plt.show()


def save_model(path: str, model_state_dict,
               epoch: int) -> None:
    """
    Saves model
    :param path: path where to save model
    """
    torch.save(
        {
            'model_state_dict': model_state_dict,
            'epoch': epoch,
        }, path)


def ignore_keyboard_traceback(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
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
                save_model(save_path, agent.state_dict(), epoch)

        if epoch % plot_every == 0:
            display.clear_output()
            print(f'Epoch #{epoch}')
            plot_progress(loss_log, reward_log)

        if epoch % refresh_target_network_freq == 0:
            target_network.load_state_dict(agent.state_dict())


@ignore_keyboard_traceback
def train_sync(n_epochs,
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

        # loss
        agent.train()
        batch_loss = 0
        for _ in range(batches_per_epoch):
            optimizer.zero_grad()
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
            agent.epsilon = 0
            reward_log.append(evaluate(1, agent, make_env(), max_steps, use_hints=use_hints))
            if best_seen_reward < reward_log[-1]:
                best_seen_reward = reward_log[-1]
                save_model(save_path, agent.state_dict(), epoch)

        if epoch % plot_every == 0:
            display.clear_output()
            print(f'Epoch #{epoch}')
            plot_progress(loss_log, reward_log)

        if epoch % refresh_target_network_freq == 0:
            target_network.load_state_dict(agent.state_dict())


def polyak_update(polyak_factor, target_network, network):
    for target_param, param in zip(target_network.parameters(), network.parameters()):
        target_param.data.copy_(polyak_factor * param.data + target_param.data * (1.0 - polyak_factor))


def train_batch(agent, target_network, optimizer, 
                exp_replay, hints, batch_size, device, 
                zero_grad=True, step=True, transitions=None):
    agent.train()
    if zero_grad:
        optimizer.zero_grad()
    coords, obs, n_completed, a, r, \
    next_coords, next_obs, next_n_completed, is_done = exp_replay.sample(batch_size)

    if transitions is not None:
        transitions.append(coords)

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
    grad_norm = nn.utils.clip_grad_norm_(agent.parameters(), 5)
    if step:
        optimizer.step()
    return loss.item(), grad_norm

@ignore_keyboard_traceback
def train_priority(n_epochs,
                   make_env,
                   agent,
                   target_network,
                   optimizer,
                   exp_replay,
                   priority_exp_replay,
                   hints,
                   epsilon,
                   trajs_per_epoch,
                   batches_per_epoch,
                   priority_batches_per_epoch,
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

    env = make_env()
    loss_log = []
    reward_log = []
    grad_norm_log = []
    init_state_v_log = []
    best_seen_reward = min_reward_for_saving

    use_hints = False
    if hints is not None:
        use_hints = True
    
    transitions = []
    priority_transitions = []
    
    for epoch in range(n_epochs + 1):
        agent.epsilon = epsilon.get_value(epoch)
        
        collect_trajectories_batch(trajs_per_epoch, agent, make_env, exp_replay, 
                                   priority_exp_replay, max_steps, use_hints=use_hints)

        # loss
        agent.train()
        batch_loss = 0
        batch_grad_norm = 0
        for _ in range(batches_per_epoch):
            loss, grad = train_batch(agent, target_network, optimizer, 
                                      exp_replay, hints, batch_size, device, 
                                      zero_grad=True, step=False, transitions=transitions)
            batch_loss += loss
            batch_grad_norm = max(batch_grad_norm, grad)

            loss, grad = train_batch(agent, target_network, optimizer, 
                                      priority_exp_replay, hints, batch_size, device,
                                      zero_grad=False, step=True, transitions=priority_transitions)
            batch_loss += loss
            batch_grad_norm = max(batch_grad_norm, grad)

        if (epoch + 1) % loss_log_freq == 0:
            batch_loss /= batches_per_epoch + priority_batches_per_epoch
            if batch_loss > loss_clip:
                loss_log.append(loss_clip)
            else:
                loss_log.append(batch_loss)

            initial_state_v = np.max(agent.get_qvalues(env.reset()))
            init_state_v_log.append(initial_state_v)
            grad_norm_log.append(batch_grad_norm)
            
        
        if (epoch + 1) % reward_log_freq == 0:
            eps = agent.epsilon
            agent.epsilon = 0
            reward_log.append(evaluate(1, agent, make_env(), max_steps, use_hints=use_hints))
            if best_seen_reward < reward_log[-1]:
                best_seen_reward = reward_log[-1]
                save_model(save_path, agent.state_dict(), epoch)
            agent.epsilon = eps

        if (epoch + 1) % plot_every == 0:
            transitions = np.vstack(transitions)
            priority_transitions = np.vstack(priority_transitions)

            display.clear_output()
            print(f'Epoch #{epoch}')
            plot_progress(loss_log, reward_log, init_state_v_log, grad_norm_log, 
                          transitions, priority_transitions, env._get_field_image())
            transitions = []
            priority_transitions = []

        if (epoch + 1) % refresh_target_network_freq == 0:
            polyak_update(0.5, target_network, agent)
            # target_network.load_state_dict(agent.state_dict())
    return reward_log

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

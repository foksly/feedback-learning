import typing as tp
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

import matplotlib.pyplot as plt
from IPython import display
from functools import wraps

# ---------------------------------------------
#         Traj collection + evaluation
# ---------------------------------------------


def hints2tensor(hints: tp.List[tp.Any],
                 hint_type: str,
                 device=torch.device('cuda')):
    if hint_type == 'next_direction':
        hints_t = torch.as_tensor(hints, device=device)
        return hints_t


def collect_trajectories(n_trajs,
                         agent,
                         make_env,
                         exp_replay,
                         priority_exp_replay,
                         max_steps=100,
                         device=torch.device('cuda')):

    agent.eval()

    env = make_env()
    envs = [make_env() for _ in range(n_trajs)]
    trajs = [[] for _ in range(n_trajs)]
    states = [env.reset() for env in envs]
    hints = [
        env.get_hint(state, hint_type=agent.hint_type) for state in states
    ]
    n_steps = 0
    dones = [False for _ in range(n_trajs)]
    rewards = [0 for _ in range(n_trajs)]

    while (n_steps < max_steps):
        n_steps += 1
        coords = []
        obses = []
        n_completed = []
        step_hints = []

        for i in range(len(dones)):
            if not dones[i]:
                coords.append(np.array(states[i][0]))
                obses.append(states[i][1])
                n_completed.append(np.array([states[i][2]]))
                step_hints.append(hints[i])
        if len(coords) == 0:
            break

        coords_t = torch.as_tensor(np.array(coords), device=device)
        obses_t = torch.as_tensor(np.array(obses),
                                  dtype=torch.long,
                                  device=device)
        n_completed_t = torch.as_tensor(np.array(n_completed), device=device)
        hints_t = hints2tensor(step_hints, hint_type=agent.hint_type)

        with torch.no_grad():
            qvalues = agent(coords_t, obses_t, n_completed_t, hints_t)

        actions = agent.sample_actions(qvalues.cpu().numpy())
        idx = 0
        for i in range(len(dones)):
            if not dones[i]:
                next_s, r, done = envs[i].step(actions[idx])
                rewards[i] += r
                trajs[i].append(
                    [states[i], actions[idx], r, next_s, done, hints[i]])
                if done:
                    dones[i] = True
                states[i] = next_s
                hints[i] = env.get_hint(next_s, hint_type=agent.hint_type)
                idx += 1
    for i in range(len(trajs)):
        exp_replay.add(trajs[i])
        priority_exp_replay.add(trajs[i], rewards[i])


def evaluate(n_trajs, agent, env, max_steps=100):
    agent.eval()
    rewards = []
    for _ in range(n_trajs):
        s = env.reset()
        total_reward = 0
        n_steps = 0
        done = False
        while (not done and n_steps < max_steps):
            hint = env.get_hint(s, hint_type=agent.hint_type)
            qvalues = agent.get_qvalues(s, hint)

            a = agent.sample_actions(qvalues).item()
            next_s, r, done = env.step(a)
            total_reward += r
            s = next_s
            n_steps += 1
        rewards.append(total_reward)
    return np.mean(rewards)


def compute_td_loss(coords,
                    obses,
                    n_completed,
                    actions,
                    rewards,
                    next_coords,
                    next_obses,
                    next_n_completed,
                    is_done,
                    hints,
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

    hints = hints2tensor(hints, hint_type=agent.hint_type)
    is_not_done = 1 - is_done

    predicted_qvalues = agent(coords, obses, n_completed, hints)
    predicted_next_qvalues_target = target_network(next_coords, next_obses,
                                                   next_n_completed, hints)
    predicted_next_qvalues_agent = agent(next_coords, next_obses,
                                         next_n_completed, hints)

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


def save_model(path: str, model_state_dict, epoch: int, reward: int) -> None:
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


def plot_progress(loss_log, reward_log):
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    axs[0].plot(loss_log)
    axs[0].set_title('TD loss history')

    axs[1].plot(reward_log)
    axs[1].set_title('Mean reward per episode')

    plt.show()


def train_batch(agent,
                target_network,
                optimizer,
                exp_replay,
                batch_size,
                device,
                zero_grad=True,
                step=True):
    agent.train()
    if zero_grad:
        optimizer.zero_grad()

    coords, obs, n_completed, a, r, \
    next_coords, next_obs, next_n_completed, \
    is_done, hints = exp_replay.sample(batch_size, add_hint=True)

    loss = compute_td_loss(coords,
                           obs,
                           n_completed,
                           a,
                           r,
                           next_coords,
                           next_obs,
                           next_n_completed,
                           is_done,
                           hints,
                           agent,
                           target_network,
                           device=device)
    loss.backward()
    if step:
        optimizer.step()
    return loss.item()

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
          make_env,
          agent,
          target_network,
          optimizer,
          exp_replay,
          priority_exp_replay,
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

    loss_log = []
    reward_log = []
    best_seen_reward = min_reward_for_saving

    for epoch in range(n_epochs + 1):
        agent.epsilon = epsilon.get_value(epoch)
        collect_trajectories(trajs_per_epoch, agent, make_env, exp_replay,
                             priority_exp_replay, max_steps)

        # loss
        agent.train()
        batch_loss = 0
        for _ in range(batches_per_epoch):
            batch_loss += train_batch(agent,
                                      target_network,
                                      optimizer,
                                      exp_replay,
                                      batch_size,
                                      device,
                                      zero_grad=True,
                                      step=False)
            batch_loss += train_batch(agent,
                                      target_network,
                                      optimizer,
                                      priority_exp_replay,
                                      batch_size,
                                      device,
                                      zero_grad=False,
                                      step=True)

        if (epoch + 1) % loss_log_freq == 0:
            batch_loss /= batches_per_epoch + priority_batches_per_epoch
            if batch_loss > loss_clip:
                loss_log.append(loss_clip)
            else:
                loss_log.append(batch_loss)

        if (epoch + 1) % reward_log_freq == 0:
            eps = agent.epsilon
            agent.epsilon = 0
            reward_log.append(evaluate(1, agent, make_env(), max_steps))
            if best_seen_reward < reward_log[-1]:
                best_seen_reward = reward_log[-1]
                save_model(save_path, agent.state_dict(), epoch,
                           best_seen_reward)
            agent.epsilon = eps

        if (epoch + 1) % plot_every == 0:
            display.clear_output()
            print(f'Epoch #{epoch}')
            plot_progress(loss_log, reward_log)

        if (epoch + 1) % refresh_target_network_freq == 0:
            target_network.load_state_dict(agent.state_dict())
    return reward_log

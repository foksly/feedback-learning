import numpy as np
import pickle
from env import Environment

with open('configs/extra_simple_field_config.pkl', 'rb') as file:
    CONFIG = pickle.load(file)


def test_env_reset_state():
    env = Environment(CONFIG, add_agent_value=False)
    s = env.reset()
    assert s.coordinate == (3, 1)
    assert np.all(s.observation == np.array([[2, 0, 0], [2, 0, 0], [2, 2, 2]]))


def test_step():
    env = Environment(CONFIG, add_agent_value=False)
    env.step('R')
    assert env.agent_position == (3, 2)

    env.step('U')
    assert env.agent_position == (2, 2)

    env.step('L')
    assert env.agent_position == (2, 1)


def test_env_reset_position_after_few_steps():
    env = Environment(CONFIG, add_agent_value=False)
    env.reset()
    env.step('R')
    env.step('U')
    assert env.agent_position != env.start_position

    env.reset()
    assert env.agent_position == env.start_position


def test_step_into_wall_does_not_change_position():
    env = Environment(CONFIG, add_agent_value=False)
    s = env.reset()
    next_s, _, _ = env.step('D')
    assert s.coordinate == next_s.coordinate
    assert np.all(s.observation == next_s.observation)

    next_s, _, _ = env.step('L')
    assert s.coordinate == next_s.coordinate
    assert np.all(s.observation == next_s.observation)

    env.step('U')
    s, _, _ = env.step('U')
    next_s, _, _ = env.step('L')
    assert s.coordinate == next_s.coordinate
    assert np.all(s.observation == next_s.observation)

    next_s, _, _ = env.step('U')
    assert s.coordinate == next_s.coordinate
    assert np.all(s.observation == next_s.observation)


def test_lava_is_terminal():
    env = Environment(CONFIG, add_agent_value=False)
    _, r, done = env.step('R')
    assert r == 0
    assert not done

    _, r, done = env.step('R')
    assert r == -10
    assert done


def test_successful_trajectory():
    env = Environment(CONFIG, add_agent_value=False)
    steps = ['R', 'U', 'R', 'R', 'R', 'R', 'R', 'U']
    for step in steps:
        s, r, done = env.step(step)

    assert s.coordinate == (1, 6)
    assert np.all(s.observation == np.array([[2, 2, 2], [0, 0, 2], [0, 0, 2]]))
    assert r == 100
    assert done


def test_reward_for_bridge_completion_is_given_only_once():
    env = Environment(CONFIG, add_agent_value=False)
    steps = ['R', 'U', 'R', 'R', 'R']
    for step in steps:
        _, r, _ = env.step(step)

    assert r == 1

    _, r, _ = env.step('L')
    _, r, _ = env.step('R')

    assert r == 0


def test_reward_for_bridge_resets_correctly():
    env = Environment(CONFIG, add_agent_value=False)
    steps = ['R', 'U', 'R', 'R', 'R']
    for step in steps:
        _, r, _ = env.step(step)

    env.reset()
    for step in steps:
        _, r, _ = env.step(step)
    assert r == 1

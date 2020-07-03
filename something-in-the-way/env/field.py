import pickle
import numpy as np


def save_field(field, name):
    with open(name, 'wb') as file:
        pickle.dump(field, file)


def load_field(name):
    with open(name, 'rb') as file:
        field = pickle.load(file)
    return field


def _build_bridge(coord, field, pattern, value):
    """
    coord is the first bridge cell
    bridge should be in format 'rdr' - right, down, right
    """
    hints = [(coord[0], coord[1] - 1, 'R')]

    field[coord[0]][coord[1]] = value
    for direction in pattern:
        hints.append((coord[0], coord[1], direction.upper()))
        if direction.upper() == 'R':
            coord[1] += 1
        elif direction.upper() == 'L':
            coord[1] -= 1
        elif direction.upper() == 'D':
            coord[0] += 1
        elif direction.upper() == 'U':
            coord[0] -= 1

        field[coord[0]][coord[1]] = value

    hints.append((coord[0], coord[1], 'R'))
    return hints


def _compute_pattern_range(pattern):
    current_y = 0
    upper_range = 0
    lower_range = 0

    for p in pattern:
        if p.upper() == 'U':
            current_y += 1
            upper_range = max(upper_range, current_y)
        elif p.upper() == 'D':
            current_y -= 1
            lower_range = min(lower_range, current_y)
    return -lower_range, upper_range


def _make_config(field, receptive_field_size, start_position, goal_position,
                value, reward, hints):
    config = {
        'field_numpy': field,
        'receptive_field_size': receptive_field_size,
        'start_position': start_position,
        'goal_position': goal_position,
        'value': value,
        'reward': reward,
        'hints': hints
    }
    return config


def field_generator(n_bridges,
                    height,
                    receptive_field=3,
                    block_size=3,
                    patterns=['rr'],
                    bridge_reward=10,
                    values={
                        'goal': 1,
                        'lava': 2,
                        'wall': 3
                    },
                    reward={
                        'goal': 10,
                        'lava': -5,
                        'wall': -0.1,
                        'other': -0.2
                    }):
    """
    Height of field > 2
    Width of lava and earth blocks = block_size
    """
    assert height > 2, "Invalid height value, heigh must be grater than 2"
    values = {key: value for key, value in values.items()}
    reward = {key: value for key, value in reward.items()}

    # create field and lava blocks
    field = np.zeros((height, (n_bridges * 2 + 1) * block_size))
    for i in range(block_size, field.shape[1] - block_size, block_size * 2):
        field[:, i:i + block_size] = values['lava']

    # padding
    pad_width = (receptive_field // 2) + 1
    field = np.pad(field, pad_width=pad_width, constant_values=values['lava'])

    # generate bridges
    hints = []
    lava_blocks_y = [
        i + pad_width for i in range(block_size, field.shape[1] - pad_width -
                                     block_size, block_size * 2)
    ]

    bridge_color = max(values.values()) + 1
    values['bridges'] = set()
    for y in lava_blocks_y:
        pattern = np.random.choice(patterns)
        lower_range, upper_range = _compute_pattern_range(pattern)

        x = np.random.randint(pad_width + upper_range,
                              field.shape[0] - pad_width - lower_range)
        bridge_hints = _build_bridge([x, y], field, pattern, bridge_color)
        hints.append(bridge_hints)

        reward[bridge_hints[-1]] = bridge_reward

        values['bridges'].add(bridge_color)
        bridge_color += 1

    start_position = (field.shape[0] - pad_width - 1, pad_width)
    goal_position = (pad_width, field.shape[1] - pad_width - 1)

    return _make_config(field, receptive_field, start_position, goal_position,
                       values, reward, hints)


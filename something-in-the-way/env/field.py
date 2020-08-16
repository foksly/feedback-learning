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


def _possible_coords_for_impassable_bridge(complete_bridge_coord, complete_bridge_pattern, field, pad_width):
    possible_coords = []
    lower_bound, upper_bound = _compute_pattern_range(complete_bridge_pattern)

    if complete_bridge_coord[0] - upper_bound - pad_width - 1 > 0:
        possible_coords.extend(range(pad_width + upper_bound, 
                                     complete_bridge_coord[0] - upper_bound - 1 - lower_bound))
    if complete_bridge_coord[0] + lower_bound <= field.shape[0] - pad_width - 1:
        possible_coords.extend(range(complete_bridge_coord[0] + lower_bound + 2 + upper_bound, 
                                     field.shape[0] - pad_width - lower_bound))
    return possible_coords


def _build_impassable_bridge(coord, field, pattern, value):
    field[coord[0]][coord[1]] = value
    for direction in pattern:
        if direction.upper() == 'R':
            coord[1] += 1
        elif direction.upper() == 'L':
            coord[1] -= 1
        elif direction.upper() == 'D':
            coord[0] += 1
        elif direction.upper() == 'U':
            coord[0] -= 1

        field[coord[0]][coord[1]] = value



def _compute_pattern_range(pattern):
    current_y = 0
    upper_bound = 0
    lower_bound = 0

    for p in pattern:
        if p.upper() == 'U':
            current_y += 1
            upper_bound = max(upper_bound, current_y)
        elif p.upper() == 'D':
            current_y -= 1
            lower_bound = min(lower_bound, current_y)
    return -lower_bound, upper_bound


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
                    add_impassable_bridges=False,
                    pad_with='lava',
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
    field = np.pad(field, pad_width=pad_width, constant_values=values[pad_with])

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
        lower_bound, upper_bound = _compute_pattern_range(pattern)

        x_bridge = np.random.randint(pad_width + upper_bound,
                              field.shape[0] - pad_width - lower_bound)
        
        if add_impassable_bridges:
            possible_start_coords = _possible_coords_for_impassable_bridge([x_bridge, y], pattern, field, pad_width)
            x_impassable = np.random.choice(possible_start_coords)

            should_build_impassable_first = np.random.randint(2)
            if should_build_impassable_first:
                x_bridge, x_impassable = x_impassable, x_bridge
            impassable_bridge_pattern = pattern[:-1]
            _build_impassable_bridge([x_impassable, y], field,
                                      impassable_bridge_pattern, bridge_color)
        
        bridge_hints = _build_bridge([x_bridge, y], field, pattern, bridge_color)
        hints.append(bridge_hints)
        reward[bridge_hints[-1]] = bridge_reward
        values['bridges'].add(bridge_color)
        bridge_color += 1

    start_position = (field.shape[0] - pad_width - 1, pad_width)
    goal_position = (pad_width, field.shape[1] - pad_width - 1)

    return _make_config(field, receptive_field, start_position, goal_position,
                       values, reward, hints)

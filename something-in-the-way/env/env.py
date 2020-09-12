import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from collections import namedtuple


Position = namedtuple('Position', ['x', 'y'])
State = namedtuple('State', ['coord', 'obs', 'n_completed'])

class Reward:
    def __init__(self, field_config):
        if isinstance(field_config['field_numpy'], str):
            self.field = np.load(field_config['field_numpy'])
        else:
            self.field = field_config['field_numpy']
        self.value = field_config['value']
        self.goal_position = Position(*field_config['goal_position'])
        self.reward_config = deepcopy(field_config['reward'])
        self.completed_bridges = 0
        self.seen_states = set()

    def reset(self):
        self.completed_bridges = 0
        self.seen_states = set()

    def __call__(self, position, action, next_position):
        if next_position == self.goal_position:
            return self.reward_config['goal']
        if self.field[next_position.x][next_position.y] == self.value['lava']:
            return self.reward_config['lava']
        if position == next_position:
            return self.reward_config['wall']
        if (*position, action) in self.reward_config and (*position, action) not in self.seen_states:
            self.completed_bridges += 1
            self.seen_states.add((*position, action))
            return self.reward_config[(*position, action)]
        if 'action' in self.reward_config and action in self.reward_config['action']:
            return self.reward_config['action'][action]
        return self.reward_config['other']

class Environment:
    LAVA_COLOR = np.array([255, 140, 0])
    WALL_COLOR = np.array([128, 128, 128])
    BRIDGE_COLOR = np.array([0, 0, 0])
    GOAL_COLOR = np.array([0, 0, 250])
    AGENT_COLOR = np.array([0, 128, 0])

    def __init__(self, field_config, add_agent_value=True):
        self.field_config = field_config
        self.add_agent_value = add_agent_value

        if isinstance(field_config['field_numpy'], str):
            self.field = np.load(field_config['field_numpy'])
        else:
            self.field = field_config['field_numpy']
        self.height = self.field.shape[0]
        self.width = self.field.shape[1]

        self.start_position = Position(*field_config['start_position'])
        self.goal_position = Position(*field_config['goal_position'])
        self.agent_position = self.start_position
        self.agent_value = max(max(v) if isinstance(v, set) 
                               else v for v in self.field_config['value'].values()) + 1

        self.receptive_field_size = field_config['receptive_field_size']
        assert self.receptive_field_size % 2, "receptive field value must be odd"
        self.padding_size = self.receptive_field_size // 2

        self.value = field_config['value']
        self.reward = Reward(field_config)

        self.hints = field_config['hints']

        self.index2action = {0: 'U', 1: 'D', 2: 'R', 3: 'L'}
        self.action2index = {v: k for k, v in self.index2action.items()}

    def step(self, action):
        if not isinstance(action, str):
            action = self.index2action[action]
        next_position = self._get_next_position(self.agent_position, action)
        is_done = False
        if next_position == self.goal_position:
            is_done = True
        if self.field[next_position.x][next_position.y] == self.value['lava']:
            is_done = True

        reward_for_step = self.reward(self.agent_position, action, next_position)
        self.agent_position = next_position
        return State(self.agent_position, 
                     self._get_receptive_field(self.agent_position), 
                     self.reward.completed_bridges), reward_for_step, is_done
    

    def get_current_state(self):
        return State(self.agent_position, 
                     self._get_receptive_field(self.agent_position),
                     self.reward.completed_bridges)
                
    def get_hint(self, state, hint_type):
        # self.action2index = {'U': 0, 'D': 1, 'R': 2}
        if hint_type.startswith('next_direction'):
            if state.n_completed < len(self.hints):
                for hint in self.hints[state.n_completed]:
                    if (hint[0], hint[1]) == state.coord:
                        correct_action = hint[2]
                        return self.action2index[correct_action]
                if hint_type == 'next_direction_on_bridge':
                    no_hint_token = 3
                    return no_hint_token

                target_x = self.hints[state.n_completed][0][0]
            else:
                target_x = self.goal_position.x

            if state.coord.x == target_x:
                return self.action2index['R']
            elif state.coord.x < target_x:
                return self.action2index['D']
            else:
                return self.action2index['U']


    def get_static_hint(self):
        hints = []
        for t in self.hints:
            coord = Position(t[0], t[1])
            obs = self._get_receptive_field(coord)
            state = State(coord, obs, t[2])
            hints.append(state)
        return hints

    def reset(self):
        self.agent_position = self.start_position
        self.agent_value = max(max(v) if isinstance(v, set) 
                               else v for v in self.field_config['value'].values()) + 1
        self.reward.reset()
        return State(self.agent_position, 
                     self._get_receptive_field(self.agent_position), 
                     self.reward.completed_bridges)


    def _get_field_image(self):
        field_img = np.zeros((*self.field.shape, 3))
        for i in range(self.field.shape[0]):
            for j in range(self.field.shape[1]):
                if self.field[i][j] == self.value['lava']:
                    field_img[i][j] = self.LAVA_COLOR
                elif self.field[i][j] == self.value['wall']:
                    field_img[i][j] = self.WALL_COLOR
                elif (i, j) == self.goal_position:
                    field_img[i][j] = self.GOAL_COLOR
                elif self.field[i][j] in self.value['bridges']:
                    field_img[i][j] = self.BRIDGE_COLOR
        return field_img


    def render(self, show_padding=False):
        field_img = self._get_field_image()
        field_img[self.agent_position[0]][self.agent_position[1]] = self.AGENT_COLOR
        if not show_padding:
            field_img = field_img[self.padding_size:-self.padding_size, 
                                  self.padding_size:-self.padding_size, :]
        plt.imshow(field_img.astype(np.int))
        return field_img


    def _get_next_position(self, position: Position, action: str) -> Position:
        assert action in {'U', 'D', 'R', 'L'}

        if (action == 'U' and position.x > 0):
            next_position = Position(position.x - 1, position.y)
            if self.field[next_position.x][next_position.y] != self.value['wall']:
                return next_position
            return position
        elif (action == 'D' and position.x < self.height - 1):
            next_position = Position(position.x + 1, position.y)
            if self.field[next_position.x][next_position.y] != self.value['wall']:
                return next_position
            return position
        elif (action == 'L' and position.y > 0):
            next_position = Position(position.x, position.y - 1)
            if self.field[next_position.x][next_position.y] != self.value['wall']:
                return next_position
            return position
        elif (action == 'R' and position.y < self.width - 1):
            next_position = Position(position.x, position.y + 1)
            if self.field[next_position.x][next_position.y] != self.value['wall']:
                return next_position
            return position
        else:
            return position

    def _get_receptive_field(self, position):
        if self.receptive_field_size == -1:
            return self.field
        x_from = position.x - self.padding_size
        x_to = position.x + self.padding_size + 1
        y_from = position.y - self.padding_size
        y_to = position.y + self.padding_size + 1

        receptive_field = np.copy(self.field[x_from:x_to, y_from:y_to])
        if self.add_agent_value:
            receptive_field[self.padding_size, self.padding_size] \
                = self.agent_value + self.reward.completed_bridges

        return receptive_field


class EnvFactory:
    def __init__(self, configs, eps=1e-1):
        self.configs = configs if isinstance(configs, list) else [configs]
        self.id = 0
        
        self._max_reward = sum(self.configs[0]['reward'].values()) - eps
    
    def __call__(self):
        return Environment(self.configs[self.id], add_agent_value=False)
    
    def switch(self):
        self.id = np.random.randint(len(self.configs))
    
    def get_max_reward(self):
        return self._max_reward

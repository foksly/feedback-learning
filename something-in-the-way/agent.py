import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class StatePreprocessor(nn.Module):
    def __init__(self, coord_range, field_values_range, n_completed,
                       coord_emb_dim, field_emb_dim, completed_emb_dim):
        super().__init__()
        self.coord_embedding = nn.Embedding(coord_range, coord_emb_dim)
        self.field_embedding = nn.Embedding(field_values_range, field_emb_dim)
        self.completed_bridges_embedding = nn.Embedding(n_completed + 1, completed_emb_dim)

    def forward(self, coords, obses, n_completed):
        """
        Parameters
        ----------
        coords: torch.tensor (batch_size, 2)
        obses: torch.tensor (batch_size, receptive_field, receptive_field)
        
        Returns
        -------
        torch.tensor (batch_size, embedding_dim * 2 + receptive_field ** 2)
        """
        batch_size = coords.shape[0]
        coords_emb = self.coord_embedding(coords).view(batch_size, -1)
        field_emb = self.field_embedding(obses.view(batch_size,-1)) \
                        .view(batch_size, -1)
        
        completed_bridges_emb = self.completed_bridges_embedding(n_completed) \
                                    .view(batch_size, -1)
        return torch.cat([coords_emb, field_emb, completed_bridges_emb], dim=1)

    def to_tensor(self, state):
        device = next(self.parameters()).device
        if isinstance(state, list):
            coord, obs, n_completed = [], [], []
            for hint in state:
                coord.append(np.array(hint.coord))
                obs.append(hint.obs)
                n_completed.append(np.array([hint.n_completed]))
            coord_t = torch.as_tensor(coord, device=device)
            obs_t = torch.as_tensor(obs, dtype=torch.long, device=device)
            n_completed_t = torch.as_tensor(n_completed, device=device)
            return coord_t, obs_t, n_completed_t
        
        else:
            coord = torch.as_tensor([state.coord], device=device)
            obs = torch.as_tensor([state.obs],
                                dtype=torch.long,
                                device=device)
            n_completed = torch.as_tensor([state.n_completed],
                                        device=device)
        return coord, obs, n_completed

class HintPreprocessor(nn.Module):
    def __init__(self, coord_range, field_values_range,
                       coord_emb_dim, field_emb_dim, 
                       action_range=4, action_emb_dim=2):
        super().__init__()
        self.encode_coord = nn.Embedding(coord_range, coord_emb_dim)
        self.encode_field = nn.Embedding(field_values_range, field_emb_dim)
        self.encode_action = nn.Embedding(action_range, action_emb_dim)

    def forward(self, coords, obses, actions):
        """
        Parameters
        ----------
        coords: torch.tensor (batch_size, 2)
        obses: torch.tensor (batch_size, receptive_field, receptive_field)
        
        Returns
        -------
        torch.tensor (batch_size, embedding_dim * 2 + receptive_field ** 2)
        """
        batch_size = coords.shape[0]
        coords_emb = self.encode_coord(coords).view(batch_size, -1)
        field_emb = self.encode_field(obses.view(batch_size,-1)).view(batch_size, -1)
        action_emb = self.encode_action(actions).view(batch_size, -1)
        return torch.cat([coords_emb, field_emb, action_emb], dim=1)
    
    def to_tensor(self, hints):
        device = next(self.parameters()).device
        coords, obs, action = [], [], []
        for hint in hints:
            coords.append(np.array(hint[0].coord, copy=False))
            obs.append(hint[0].obs)
            action.append(np.array([hint[1]]))
        coords_t = torch.as_tensor(coords, device=device)
        obs_t = torch.as_tensor(obs, dtype=torch.long, device=device)
        action_t = torch.as_tensor(action, device=device)
        return coords_t, obs_t, action_t



class DynamicHintDQNAgent(nn.Module):
    def __init__(self,
                 coord_range,
                 field_values_range,
                 n_completed,
                 coord_emb_dim,
                 field_emb_dim,
                 completed_emb_dim,
                 receptive_field=3,
                 n_actions=4,
                 epsilon=0.5):
        super().__init__()
        self.hint_type = 'next_direction'
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.preprocessor = StatePreprocessor(coord_range, field_values_range, n_completed,
                                              coord_emb_dim, field_emb_dim, completed_emb_dim)
        self.hint_preprocessor = nn.Embedding(n_actions, 4)

        input_dim = coord_emb_dim * 2 + \
                    receptive_field * receptive_field * field_emb_dim + \
                    completed_emb_dim + 4

        self.net = nn.Sequential(nn.Linear(input_dim, 256), nn.LeakyReLU(),
                                 nn.Linear(256, 128), nn.LeakyReLU(),
                                 nn.Linear(128, 128), nn.LeakyReLU(),
                                 nn.Linear(128, n_actions))

    def forward(self, coords, obses, n_completed, hints):
        """
        takes agent's observation (tensor), returns qvalues (tensor)
        """
        states = self.preprocessor(coords, obses, n_completed)
        hints = self.hint_preprocessor(hints)

        features = torch.cat([states, hints], dim=1)
        qvalues = self.net(features)

        return qvalues

    def get_qvalues(self, state, hint):
        """
        qvalue for a single raw state
        """
        coord, obs, n_completed = self.preprocessor.to_tensor(state)
        hint = torch.as_tensor([hint], device=next(self.parameters()).device)

        with torch.no_grad():
            qvalues = self.forward(coord, obs, n_completed, hint)

        return qvalues.cpu().numpy()

    def sample_actions(self, qvalues, greedy=True):
        """ Pick actions given qvalues. Uses epsilon-greedy exploration strategy. """
        epsilon = self.epsilon
        batch_size, n_actions = qvalues.shape

        random_actions = np.random.choice(n_actions, size=batch_size)
        best_actions = qvalues.argmax(axis=-1)

        should_explore = np.random.choice([0, 1],
                                          batch_size,
                                          p=[1 - epsilon, epsilon])
        return np.where(should_explore, random_actions, best_actions)


class DQNAgent(nn.Module):
    def __init__(self,
                 coord_range,
                 field_values_range,
                 n_completed,
                 coord_emb_dim,
                 field_emb_dim,
                 completed_emb_dim,
                 receptive_field=3,
                 n_actions=4,
                 epsilon=0.5):
        super().__init__()
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.preprocessor = StatePreprocessor(coord_range, field_values_range, n_completed,
                                              coord_emb_dim, field_emb_dim, completed_emb_dim)
        input_dim = coord_emb_dim * 2 + \
                    receptive_field * receptive_field * field_emb_dim + \
                    completed_emb_dim

        self.net = nn.Sequential(nn.Linear(input_dim, 256), nn.LeakyReLU(),
                                 nn.Linear(256, 128), nn.LeakyReLU(),
                                 nn.Linear(128, 128), nn.LeakyReLU(),
                                 nn.Linear(128, n_actions))

    def forward(self, coords, obses, n_completed):
        """
        takes agent's observation (tensor), returns qvalues (tensor)
        """
        states = self.preprocessor(coords, obses, n_completed)
        qvalues = self.net(states)

        return qvalues

    def get_qvalues(self, state):
        """
        qvalue for a single raw state
        """
        coord, obs, n_completed = self.preprocessor.to_tensor(state)
        with torch.no_grad():
            qvalues = self.forward(coord, obs, n_completed)
        return qvalues.cpu().numpy()

    def sample_actions(self, qvalues, greedy=True):
        """ Pick actions given qvalues. Uses epsilon-greedy exploration strategy. """
        epsilon = self.epsilon
        batch_size, n_actions = qvalues.shape

        random_actions = np.random.choice(n_actions, size=batch_size)
        best_actions = qvalues.argmax(axis=-1)

        should_explore = np.random.choice([0, 1],
                                          batch_size,
                                          p=[1 - epsilon, epsilon])
        return np.where(should_explore, random_actions, best_actions)


class DotAttention(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, state, hints):
        attn_scores = torch.matmul(state, hints.T) / state.shape[1]
        attn_weights = F.softmax(attn_scores, dim=1).unsqueeze(2)
        attn_vec = torch.sum(hints * attn_weights, dim=1)
        return attn_vec


class AdditiveAttention(nn.Module):
    def __init__(self, attn_dim):
        super().__init__()
        self.additive_attn_scores = nn.Linear(attn_dim, 1)
    
    def forward(self, state, hints):
        attn_scores = self.additive_attn_scores(torch.tanh(hints + state.unsqueeze(1)))
        attn_weights = F.softmax(attn_scores, dim=1)
        attn_vec = (hints * attn_weights).sum(1)
        return attn_vec


class FCAttention(nn.Module):
    def __init__(self, attn_dim, state_dim, score='dot'):
        super().__init__()
        self.encode = nn.Linear(state_dim, attn_dim)
        
        assert score in {'dot', 'additive'}, "score must be either 'dot' or 'additive'"
        self.attention = DotAttention() if score == 'dot' else AdditiveAttention(attn_dim)
    
    def forward(self, state, hints):
        s_enc = self.encode(state)
        h_enc = self.encode(hints)
        attn_vec = self.attention(s_enc, h_enc)
        return attn_vec


class LSTMAttention(nn.Module):
    def __init__(self, attn_dim, state_dim, hidden_dim, score='dot'):
        super().__init__()
        self.encode = nn.Linear(state_dim, attn_dim)
        self.lstm = nn.LSTM(state_dim, hidden_dim, bidirectional=True)
        self.encode_hint = nn.Linear(hidden_dim * 2, attn_dim)
        
        assert score in {'dot', 'additive'}, "score must be either 'dot' or 'additive'"
        self.attention = DotAttention() if score == 'dot' else AdditiveAttention(attn_dim)
    
    def forward(self, state, hints):
        s_enc = self.encode_state(state)
        h_enc, _ = self.lstm(hints.unsqueeze(1))
        h_enc = self.encode_hint(h_enc.squeeze(1))
        attn_vec = self.attention(s_enc, h_enc)
        return attn_vec


class TransformerAttention(nn.Module):
    def __init__(self, attn_dim, state_dim,
                       d_model, nhead, num_layers):
        super().__init__()
        self.encode = nn.Linear(state_dim, attn_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.t_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead)
        self.t_decoder = nn.TransformerDecoder(s_enc.unsqueeze(0), h_self_attn)
        return s_decoded.squeeze(0)


class DQNAttnAgent(nn.Module):
    def __init__(self,
                 coord_range,
                 field_values_range,
                 n_completed,
                 coord_emb_dim,
                 field_emb_dim,
                 completed_emb_dim,
                 attn_dim,
                 attn_model='fc',
                 receptive_field=3,
                 n_actions=4,
                 epsilon=0.5,
                 **kwargs):
        super().__init__()
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.preprocessor = StatePreprocessor(coord_range, field_values_range, n_completed,
                                                    coord_emb_dim, field_emb_dim, completed_emb_dim)
        
        state_dim = coord_emb_dim * 2 + \
                    receptive_field * receptive_field * field_emb_dim + \
                    completed_emb_dim
        
        assert attn_model in {'fc', 'lstm', 'transformer'}, "attn_model should be in {'fc', 'lstm', 'transformer'}"

        if attn_model == 'fc':
            self.attention = FCAttention(attn_dim, state_dim, score=kwargs['attn_score'])
        elif attn_model == 'lstm':
            self.attention = LSTMAttention(attn_dim, state_dim, kwargs['hidden_dim'], score=kwargs['attn_score'])
        elif attn_model == 'transformer':
            self.attention = TransformerAttention(attn_dim, state_dim,
                                                  kwargs['d_model'], kwargs['nhead'], kwargs['num_layers'])

        self.net = nn.Sequential(nn.Linear(state_dim + attn_dim, 256), nn.LeakyReLU(),
                                 nn.Linear(256, 128), nn.LeakyReLU(),
                                 nn.Linear(128, 128), nn.LeakyReLU(),
                                 nn.Linear(128, n_actions))
        # self.net = nn.Linear(state_dim + attn_dim, n_actions)
    
    def forward(self, coords, obses, n_completed,
                      hint_coords, hint_obses, hint_n_completed):
        """
        takes agent's observation (tensor), returns qvalues (tensor)
        """
        states = self.preprocessor(coords, obses, n_completed)
        hints = self.preprocessor(hint_coords, hint_obses, hint_n_completed)
        attn_vec = self.attention(states, hints)
        qvalues = self.net(torch.cat([states, attn_vec], dim=1))

        return qvalues

    def get_qvalues(self, state, hints):
        """
        qvalue for a single raw state
        """
        coord, obs, n_completed = self.preprocessor.to_tensor(state)
        hint_coord, hint_obs, hint_n_completed = self.preprocessor.to_tensor(hints)
        qvalues = self.forward(coord, obs, n_completed, hint_coord, hint_obs, hint_n_completed)
        return qvalues.detach().cpu().numpy()

    def sample_actions(self, qvalues, greedy=True):
        """ Pick actions given qvalues. Uses epsilon-greedy exploration strategy. """
        epsilon = self.epsilon
        batch_size, n_actions = qvalues.shape

        random_actions = np.random.choice(n_actions, size=batch_size)
        best_actions = qvalues.argmax(axis=-1)

        should_explore = np.random.choice([0, 1],
                                          batch_size,
                                          p=[1 - epsilon, epsilon])
        return np.where(should_explore, random_actions, best_actions)

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class StateEncoder(nn.Module):
    def __init__(self, state_kwargs, receptive_field=3):
        super().__init__()
        self.n_cols = state_kwargs['n_cols'] # number of field columns
        self.x_coord_embedder = nn.Embedding(state_kwargs['coord_range'], state_kwargs['coord_emb_dim'])
        self.y_coord_embedder = nn.Embedding(state_kwargs['coord_range'], state_kwargs['coord_emb_dim'])
        self.field_embedder = nn.Embedding(state_kwargs['field_values_range'], state_kwargs['field_emb_dim'])
        self.completed_bridges_embedder = nn.Embedding(state_kwargs['n_completed'] + 1, state_kwargs['completed_emb_dim'])
        self.state_dim = 2 * state_kwargs['coord_emb_dim'] + receptive_field * receptive_field * state_kwargs['field_emb_dim'] + \
                         state_kwargs['completed_emb_dim']

    def forward(self, states):
        """
        Parameters
        ----------
        coords: torch.tensor (batch_size, 2)
        obses: torch.tensor (batch_size, receptive_field, receptive_field)
        
        Returns
        -------
        torch.tensor (batch_size, embedding_dim * 2 + receptive_field ** 2)
        """
        coords, obs, n_completed = self.to_tensor(states)

        batch_size = coords.shape[0]
        # coords = coords[:, 0] * self.n_cols + coords[:, 1]

        x_coord_emb = self.x_coord_embedder(coords[:, 0]).view(batch_size, -1)
        y_coord_emb = self.y_coord_embedder(coords[:, 1]).view(batch_size, -1)

        field_emb = self.field_embedder(obs.view(batch_size,-1)) \
                        .view(batch_size, -1)

        completed_bridges_emb = self.completed_bridges_embedder(n_completed) \
                                    .view(batch_size, -1)
        return torch.cat([x_coord_emb, y_coord_emb, field_emb, completed_bridges_emb], dim=1)

    def to_tensor(self, states):
        """
        Convert single state or list of states into three tensors:
        coordinates, observations, number of completed bridges
        """
        device = next(self.parameters()).device
        if not isinstance(states, list):
            states = [states]
        
        coords, obs, n_completed = [], [], []
        for state in states:
            coords.append(state.coord)
            obs.append(state.obs)
            n_completed.append(state.n_completed)
        
        coord_t = torch.as_tensor(coords, device=device)
        obs_t = torch.as_tensor(obs, dtype=torch.long, device=device)
        n_completed_t = torch.as_tensor(n_completed, device=device)
        return coord_t, obs_t, n_completed_t


class HintEncoder(nn.Module):
    """
    Encodes hints for further pass into a neural network

    Hint types:
    - "next_direction": direction of the correct move from a given state
      :type: int
      :hint_kwargs:
        - hint_values_range: number of different possible values of the hint
        - embedding_dim
    - "next_direction_on_bridge": almost the same as the previous one but hint
                                  if given only on bridges otherwise returns
                                  <no hint token>
    """
    def __init__(self, hint_type, hint_kwargs):
        super().__init__()
        if hint_type in {'next_direction', 'next_direction_on_bridge'}:
            self.encoder = NextDirectionHintEncoder(hint_kwargs)

        self.hint_dim = self.encoder.hint_dim

    def forward(self, hint):
        return self.encoder(hint)


class NextDirectionHintEncoder(nn.Module):
    def __init__(self, hint_kwargs):
        super().__init__()
        self.encode = nn.Embedding(hint_kwargs['hint_values_range'], hint_kwargs['embedding_dim'])
        self.hint_dim = hint_kwargs['embedding_dim']

    def forward(self, hint):
        hint_t = self.to_tensor(hint)
        return self.encode(hint_t)

    def to_tensor(self, hint):
        device = next(self.parameters()).device

        if not isinstance(hint, list):
            hint = [hint]
        hint_t = torch.as_tensor(hint, device=device)
        return hint_t


class CoordinatesAttentionHintEncoder(nn.Module):
    def __init__(self, hint_kwargs):
        super().__init__()
        self.encode_x = nn.Embedding(hint_kwargs['hint_values_range'], hint_kwargs['embedding_dim'])
        self.encode_y = nn.Embedding(hint_kwargs['hint_values_range'], hint_kwargs['embedding_dim'])
        self.hint_dim = hint_kwargs['embedding_dim']
    
    def forward(self, hint):
        pass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class StatePreprocessor(nn.Module):
    def __init__(self, coord_range, field_values_range, coord_emb_dim,
                 field_emb_dim):
        super().__init__()
        self.coord_embedding = nn.Embedding(coord_range, coord_emb_dim)
        self.field_embedding = nn.Embedding(field_values_range, field_emb_dim)

    def forward(self, coords, obses):
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
        field_emb = self.field_embedding(obses.view(batch_size,
                                                    -1)).view(batch_size, -1)
        return torch.cat([coords_emb, field_emb], dim=1)

    def to_tensor(self, state):
        device = next(self.parameters()).device
        coord = torch.as_tensor([state.coordinate], device=device)
        obs = torch.as_tensor([state.observation],
                              dtype=torch.long,
                              device=device)
        return coord, obs


class DQNAgent(nn.Module):
    def __init__(self,
                 coord_range,
                 field_values_range,
                 coord_emb_dim,
                 field_emb_dim,
                 receptive_field=3,
                 n_actions=4,
                 epsilon=0.5):
        super().__init__()
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.preprocessor = StatePreprocessor(coord_range, field_values_range,
                                              coord_emb_dim, field_emb_dim)
        input_dim = coord_emb_dim * 2 + receptive_field * receptive_field * field_emb_dim

        self.net = nn.Sequential(nn.Linear(input_dim, 256), nn.LeakyReLU(),
                                 nn.Linear(256, 128), nn.LeakyReLU(),
                                 nn.Linear(128, n_actions))

    def forward(self, coords, obses):
        """
        takes agent's observation (tensor), returns qvalues (tensor)
        """
        states = self.preprocessor(coords, obses)
        qvalues = self.net(states)
        assert qvalues.requires_grad, "qvalues must be a torch tensor with grad"

        return qvalues

    def get_qvalues(self, state):
        """
        qvalue for a single raw state
        """
        coord, obs = self.preprocessor.to_tensor(state)
        qvalues = self.forward(coord, obs)
        return qvalues.data.cpu().numpy()

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
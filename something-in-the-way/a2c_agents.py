import typing as tp

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from embedders import StateEncoder, HintEncoder
from utils import ignore_keyboard_traceback, save_model

import matplotlib.pyplot as plt
from IPython import display


class A2CAgent(nn.Module):
    def __init__(self,
                 state_config,
                 hint_type=None,
                 hint_config=None,
                 receptive_field=3,
                 n_actions=4,
                 epsilon=0.1):
        super().__init__()
        self.n_actions = n_actions
        self.hint_type = hint_type

        self.state_encoder = StateEncoder(state_config,
                                          receptive_field=receptive_field)
        hint_dim = 0
        if hint_type is not None:
            self.hint_encoder = HintEncoder(hint_type, hint_config)
            hint_dim = self.hint_encoder.dim

        self.backbone = nn.Sequential(
            nn.Linear(self.state_encoder.dim + hint_dim, 512),
            nn.LeakyReLU(), nn.Linear(512, 256), nn.LeakyReLU(),
            nn.Linear(256, 128), nn.LeakyReLU())
        self.policy_head = nn.Sequential(nn.Linear(128, 64), nn.ELU(),
                                         nn.Linear(64, n_actions))
        self.value_head = nn.Sequential(nn.Linear(128, 64), nn.ELU(),
                                        nn.Linear(64, 1))

    def forward(self, states, hint=None):
        backbone_input = self.state_encoder(states)
        if hint is not None:
            hint_enc = self.hint_encoder(hint)
            backbone_input = torch.cat([backbone_input, hint_enc], dim=1)  #

        backbone = self.backbone(backbone_input)
        logps = F.log_softmax(self.policy_head(backbone), -1)
        values = self.value_head(backbone)
        return logps, values

    def sample_actions(self, logps):
        actions = []
        probs = torch.exp(logps).detach().cpu().numpy()
        for p in probs:
            actions.append(np.random.choice(self.n_actions, p=p))
        return actions


class Attention(nn.Module):
    def __init__(self, state_dim, hint_dim, attn_dim, attn_type='cosine'):
        super().__init__()

        self.state_projector = nn.Linear(state_dim, attn_dim)
        self.hint_projector = nn.Linear(hint_dim, attn_dim)
        self.attn_type = attn_type

        if attn_type == 'bahdanau':
            self.attn = nn.Linear(attn_dim * 2, attn_dim)
            self.v = nn.Linear(attn_dim, 1)
        
        if attn_type == 'additive':
            self.attn = nn.Linear(attn_dim, 1)

    def forward(self, state_emb, hints_emb):
        """
        state_emb: (batch_size, state_dim)
        hints_emb: (batch_size, n_hints, hints_dim) or (n_hints, hints_dim)
        """
        if len(hints_emb.shape) == 2:
            hints_emb = hints_emb.unsqueeze(0).repeat(state_emb.shape[0], 1, 1)

        state_projected = self.state_projector(state_emb)
        hints_projected = self.hint_projector(hints_emb)

        compute_scores = getattr(self, self.attn_type + '_attn')
        scores = compute_scores(state_projected, hints_projected)
        if len(scores.shape) == 1:
            scores = scores.unsqueeze(0)
        self.attn_weights = F.softmax(scores, dim=1)
        output = (hints_emb * self.attn_weights.unsqueeze(2)).sum(1)
        return output

    
    def cosine_attn(self, state_projected, hints_projected):
        """
        state_emb: (batch_size, attn_dim)
        hints_emb: (batch_size, n_hints, attn_dim)
        """
        compute_cosine_sim = nn.CosineSimilarity(dim=2)
        return compute_cosine_sim(state_projected.unsqueeze(1), hints_projected).squeeze()

    def dot_attn(self, state_projected, hints_projected):
        return torch.bmm(state_projected.unsqueeze(1), hints_projected.permute(0, 2, 1)).squeeze()
    
    def bahdanau_attn(self, state_projected, hints_projected):
        repeated_state = state_projected.unsqueeze(1).repeat(1, hints_projected.shape[1], 1)
        concatenation = torch.cat([repeated_state, hints_projected], dim=2)
        scores = self.v(torch.tanh(self.attn(concatenation))).squeeze()
        return scores

    def additive_attn(self, state_projected, hints_projected):
        return self.attn(torch.tanh(state_projected.unsqueeze(1) + hints_projected)).squeeze()



class A2CAttnAgent(nn.Module):
    def __init__(self,
                 state_config,
                 hint_type,
                 hint_config,
                 attn_dim,
                 attn_type='cosine',
                 receptive_field=3,
                 n_actions=4,
                 epsilon=0.1):
        """
        Hint types:
        'full_trajectory'
        """
        super().__init__()
        self.n_actions = n_actions
        self.hint_type = hint_type

        self.state_encoder = StateEncoder(state_config, receptive_field=receptive_field)
        self.hint_encoder = HintEncoder(hint_type, hint_config)
        self.attn_dim = attn_dim

        self.attn = Attention(self.state_encoder.dim, self.hint_encoder.dim, attn_dim, attn_type=attn_type)
        self.backbone = nn.Sequential(
            nn.Linear(self.state_encoder.dim + attn_dim, 512),
            nn.LeakyReLU(), nn.Linear(512, 256), nn.LeakyReLU(),
            nn.Linear(256, 128), nn.LeakyReLU()
        )
        self.policy_head = nn.Sequential(nn.Linear(128, 64), nn.ELU(),
                                         nn.Linear(64, n_actions))
        self.value_head = nn.Sequential(nn.Linear(128, 64), nn.ELU(),
                                        nn.Linear(64, 1))

    def forward(self, states, hints):
        states_enc = self.state_encoder(states)
        hints_enc = self.hint_encoder(hints)

        weighted_hints = self.attn(states_enc, hints_enc)
        backbone_input = torch.cat([states_enc, weighted_hints], dim=1)

        backbone = self.backbone(backbone_input)
        logps = F.log_softmax(self.policy_head(backbone), -1)
        values = self.value_head(backbone)
        return logps, values

    def sample_actions(self, logps):
        actions = []
        probs = torch.exp(logps).detach().cpu().numpy()
        for p in probs:
            actions.append(np.random.choice(self.n_actions, p=p))
        return actions
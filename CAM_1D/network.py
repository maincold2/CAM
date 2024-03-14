# import re
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SinusoidalEncoder(nn.Module):
    """Sinusoidal Positional Encoder used in Nerf."""

    def __init__(self, x_dim, min_deg, max_deg, use_identity: bool = True):
        super().__init__()
        self.x_dim = x_dim
        self.min_deg = min_deg
        self.max_deg = max_deg
        self.use_identity = use_identity
        self.register_buffer(
            "scales", torch.tensor([2**i for i in range(min_deg, max_deg)])
        )

    @property
    def latent_dim(self) -> int:
        return (
            int(self.use_identity) + (self.max_deg - self.min_deg) * 2
        ) * self.x_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [..., x_dim]
        Returns:
            latent: [..., latent_dim]
        """
        if self.max_deg == self.min_deg:
            return x
        xb = torch.reshape(
            (x[Ellipsis, None, :] * self.scales[:, None]),
            list(x.shape[:-1]) + [(self.max_deg - self.min_deg) * self.x_dim],
        )
        latent = torch.sin(torch.cat([xb, xb + 0.5 * math.pi], dim=-1))
        if self.use_identity:
            latent = torch.cat([x] + [latent], dim=-1)
        return latent

class CAM_PE(nn.Module):
    def __init__(self, in_feats, hidden_feats, n_hidden_layers, out_feats):
        super(CAM_PE, self).__init__()

        self.time_encoder = SinusoidalEncoder(1, 0, 16, True)
        self.net = [nn.Linear(self.time_encoder.latent_dim, hidden_feats)] + \
                   [nn.Linear(hidden_feats, hidden_feats) for _ in range(n_hidden_layers)]
        self.net_last = nn.Linear(hidden_feats, out_feats)
        
        self.net = nn.ModuleList(self.net)
        
        self.affine = nn.ParameterList()
        for i in range(n_hidden_layers+1):
            self.affine.append(torch.nn.Parameter(torch.cat([torch.ones((1, 1, 64, 1)), torch.zeros((1, 1, 64, 1))],dim=1)))
        
    
    def forward(self, x):
        coord = x.clone()*2-1.0
        coord = torch.cat((torch.zeros_like(coord),coord), dim=-1)
        x = self.time_encoder(x)
        for i, l in enumerate(self.net):
            x = l(x)
            affine = F.grid_sample(self.affine[i], coord.view(1,-1,1,2), align_corners=True).view(2,-1,1)
            x = affine[0]*x + affine[1]
        x = self.net_last(x)
        return x

class CAM(nn.Module):
    def __init__(self, in_feats, hidden_feats, n_hidden_layers, out_feats):
        super(CAM, self).__init__()

        self.net = [nn.Linear(1, hidden_feats)] + \
                   [nn.Linear(hidden_feats, hidden_feats) for _ in range(n_hidden_layers)]
        self.net_last = nn.Linear(hidden_feats, out_feats)    
        self.net = nn.ModuleList(self.net)
        
        self.affine = nn.ParameterList()
        for i in range(n_hidden_layers+1):
            self.affine.append(torch.nn.Parameter(torch.cat([torch.ones((1, 1, 64, 1)), torch.zeros((1, 1, 64, 1))],dim=1)))
        
    
    def forward(self, x):
        coord = x.clone()*2-1.0
        coord = torch.cat((torch.zeros_like(coord),coord), dim=-1)
        for i, l in enumerate(self.net):
            x = l(x)
            affine = F.grid_sample(self.affine[i], coord.view(1,-1,1,2), align_corners=True).view(2,-1,1)
            x = affine[0]*x + affine[1]
        x = self.net_last(x)
        return x
    
class PE(nn.Module):
    def __init__(self, in_feats, hidden_feats, n_hidden_layers, out_feats):
        super(PE, self).__init__()

        self.time_encoder = SinusoidalEncoder(1, 0, 16, True)
        self.net = [nn.Linear(self.time_encoder.latent_dim, hidden_feats)] + \
                   [nn.Linear(hidden_feats, hidden_feats) for _ in range(n_hidden_layers)]
        self.net_last = nn.Linear(hidden_feats, out_feats)
        self.net = nn.ModuleList(self.net)
        
    
    def forward(self, x):
        x = self.time_encoder(x)
        for i, l in enumerate(self.net):
            x = l(x)
        x = self.net_last(x)
        return x
    
class NoEnc(nn.Module):
    def __init__(self, in_feats, hidden_feats, n_hidden_layers, out_feats):
        super(NoEnc, self).__init__()

        self.net = [nn.Linear(1, hidden_feats)] + \
                   [nn.Linear(hidden_feats, hidden_feats) for _ in range(n_hidden_layers)]
        self.net_last = nn.Linear(hidden_feats, out_feats)
        self.net = nn.ModuleList(self.net)
        
    
    def forward(self, x):
        for i, l in enumerate(self.net):
            x = l(x)
        x = self.net_last(x)
        return x
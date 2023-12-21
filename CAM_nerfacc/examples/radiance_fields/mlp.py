"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""

import functools
import math
from typing import Callable, Optional, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from siren_pytorch import SirenNet
import tinycudann as tcnn
from einops import rearrange
import numpy as np

def contract_to_unisphere(
    x: torch.Tensor,
    aabb: torch.Tensor,
    ord: Union[str, int] = 2,
    #  ord: Union[float, int] = float("inf"),
    eps: float = 1e-6,
    derivative: bool = False,
):
    aabb_min, aabb_max = torch.split(aabb, 3, dim=-1)
    x = (x - aabb_min) / (aabb_max - aabb_min)
    x = x * 2 - 1  # aabb is at [-1, 1]
    mag = torch.linalg.norm(x, ord=ord, dim=-1, keepdim=True)
    mask = mag.squeeze(-1) > 1

    if derivative:
        dev = (2 * mag - 1) / mag**2 + 2 * x**2 * (
            1 / mag**3 - (2 * mag - 1) / mag**4
        )
        dev[~mask] = 1.0
        dev = torch.clamp(dev, min=eps)
        return dev
    else:
        x[mask] = (2 - 1 / mag[mask]) * (x[mask] / mag[mask])
        x = x / 4 + 0.5  # [-inf, inf] is at [0, 1]
        return x

class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,  # The number of input tensor channels.
        output_dim: int = None,  # The number of output tensor channels.
        net_depth: int = 8,  # The depth of the MLP.
        net_width: int = 256,  # The width of the MLP.
        skip_layer: int = 4,  # The layer to add skip layers to.
        hidden_init: Callable = nn.init.xavier_uniform_,
        hidden_activation: Callable = nn.ReLU(),
        output_enabled: bool = True,
        output_init: Optional[Callable] = nn.init.xavier_uniform_,
        output_activation: Optional[Callable] = nn.Identity(),
        bias_enabled: bool = True,
        bias_init: Callable = nn.init.zeros_,
        coord_dim: int = 0
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.net_depth = net_depth
        self.net_width = net_width
        self.skip_layer = skip_layer
        self.hidden_init = hidden_init
        self.hidden_activation = hidden_activation
        self.output_enabled = output_enabled
        self.output_init = output_init
        self.output_activation = output_activation
        self.bias_enabled = bias_enabled
        self.bias_init = bias_init
        
        self.coord_dim = coord_dim
    
        
        self.matMode = [[0,1], [0,2], [1,2]]
        self.vecMode =  [2, 1, 0]

        self.affines = nn.ParameterList()
        
        self.hidden_layers = nn.ModuleList()
        in_features = self.input_dim
        for i in range(self.net_depth):
            self.hidden_layers.append(
                nn.Linear(in_features, self.net_width, bias=bias_enabled)
            )
            if self.coord_dim == 1:
                self.affines.append(torch.nn.Parameter(torch.cat([torch.ones((1, 1, 5, 1)), torch.zeros((1, 1, 5, 1))],dim=1)))
            elif self.coord_dim == 2:
                self.affines.append(torch.nn.Parameter(torch.cat([torch.ones((1, 1, 10, 3)), torch.zeros((1, 1, 10, 3))],dim=1)))
            elif self.coord_dim == 3:
                self.affines.append(torch.nn.Parameter(torch.cat([torch.ones((3, 1, 512, 512)), torch.zeros((3, 1, 512, 512))],dim=0)))
            
            if (
                (self.skip_layer is not None)
                and (i % self.skip_layer == 0)
                and (i > 0)
            ):
                in_features = self.net_width + self.input_dim
            else:
                in_features = self.net_width
        if self.output_enabled:
            self.output_layer = nn.Linear(
                in_features, self.output_dim, bias=bias_enabled
            )
        else:
            self.output_dim = in_features

        self.initialize()

    def initialize(self):
        def init_func_hidden(m):
            if isinstance(m, nn.Linear):
                if self.hidden_init is not None:
                    self.hidden_init(m.weight)
                if self.bias_enabled and self.bias_init is not None:
                    self.bias_init(m.bias)

        self.hidden_layers.apply(init_func_hidden)
        if self.output_enabled:

            def init_func_output(m):
                if isinstance(m, nn.Linear):
                    if self.output_init is not None:
                        self.output_init(m.weight)
                    if self.bias_enabled and self.bias_init is not None:
                        self.bias_init(m.bias)

            self.output_layer.apply(init_func_output)

    def norm_(self, x, mean, var):
        return (x-mean) / (var + 1e-6)
    
    def forward(self, x, coord=None):
        inputs = x
        for i in range(self.net_depth):
            x = self.hidden_layers[i](x)
            if coord is not None:
                if coord.shape[-1]==1:
                    coord_ = torch.cat((torch.zeros_like(coord),coord), dim=-1)
                    affine = F.grid_sample(self.affines[i], coord_.view(1,-1,1,2), align_corners=True).view(2,-1,1)
                    x = affine[0]*x + affine[1]
                elif coord.shape[-1]==2:
                    affine = F.grid_sample(self.affines[i], coord.view(1,1,-1,2), align_corners=True).view(2,-1,1)
                    x = affine[0]*x + affine[1]
                elif coord.shape[-1]==3:
                    coordinate_plane = torch.stack((coord[..., self.matMode[0]], coord[..., self.matMode[1]], coord[..., self.matMode[2]])).detach().view(3, -1, 1, 2).repeat(2,1,1,1)
                    plane_feats = F.grid_sample(self.affines[i], coordinate_plane, align_corners=True)
                    gamma = torch.prod(plane_feats[0:3],dim=0,keepdim=False).view(-1,1)
                    beta = torch.prod(plane_feats[3:],dim=0,keepdim=False).view(-1,1)
                    x = gamma*x + beta
            
            x = self.hidden_activation(x)
            if (
                (self.skip_layer is not None)
                and (i % self.skip_layer == 0)
                and (i > 0)
            ):
                x = torch.cat([x, inputs], dim=-1)
        if self.output_enabled:
            x = self.output_layer(x)
            x = self.output_activation(x)
        return x


class DenseLayer(MLP):
    def __init__(self, input_dim, output_dim, **kwargs):
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            net_depth=0,  # no hidden layers
            **kwargs,
        )


class NerfMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,  # The number of input tensor channels.
        condition_dim: int,  # The number of condition tensor channels.
        net_depth: int = 8,  # The depth of the MLP.
        net_width: int = 256,  # The width of the MLP.
        skip_layer: int = 4,  # The layer to add skip layers to.
        net_depth_condition: int = 1,  # The depth of the second part of MLP.
        net_width_condition: int = 128,  # The width of the second part of MLP.
        coord_dim: int = 0, # coordinate dimension for CAM
    ):
        super().__init__()
        self.base = MLP(
            input_dim=input_dim,
            net_depth=net_depth,
            net_width=net_width,
            skip_layer=skip_layer,
            output_enabled=False,
            coord_dim=coord_dim,
        )
        hidden_features = self.base.output_dim
        self.sigma_layer = DenseLayer(hidden_features, 1)
        if condition_dim > 0:
            self.bottleneck_layer = DenseLayer(hidden_features, net_width)
            self.rgb_layer = MLP(
                input_dim=net_width + condition_dim,
                output_dim=3,
                net_depth=net_depth_condition,
                net_width=net_width_condition,
                skip_layer=None,
            )
        else:
            self.rgb_layer = DenseLayer(hidden_features, 3)

    def query_density(self, x, coord=None):
        x = self.base(x, coord)
        raw_sigma = self.sigma_layer(x)
        return raw_sigma

    def forward(self, x, condition=None, coord=None):
        x = self.base(x, coord)
        raw_sigma = self.sigma_layer(x)
        
        if condition is not None:
            if condition.shape[:-1] != x.shape[:-1]:
                num_rays, n_dim = condition.shape
                condition = condition.view(
                    [num_rays] + [1] * (x.dim() - condition.dim()) + [n_dim]
                ).expand(list(x.shape[:-1]) + [n_dim])
            bottleneck = self.bottleneck_layer(x)
            x = torch.cat([bottleneck, condition], dim=-1)
        raw_rgb = self.rgb_layer(x, None)
        return raw_rgb, raw_sigma


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

class VanillaNeRFRadianceField(nn.Module):
    def __init__(
        self,
        net_depth: int = 8,  # The depth of the MLP.
        net_width: int = 256,  # The width of the MLP.
        skip_layer: int = 4,  # The layer to add skip layers to.
        net_depth_condition: int = 1,  # The depth of the second part of MLP.
        net_width_condition: int = 128,  # The width of the second part of MLP.
        coord_dim: int = 3,
    ) -> None:
        super().__init__()
        self.posi_encoder = SinusoidalEncoder(3, 0, 10, True)
        self.view_encoder = SinusoidalEncoder(3, 0, 4, True)
        self.mlp = NerfMLP(
            input_dim=self.posi_encoder.latent_dim,
            condition_dim=self.view_encoder.latent_dim,
            net_depth=net_depth,
            net_width=net_width,
            skip_layer=skip_layer,
            net_depth_condition=net_depth_condition,
            net_width_condition=net_width_condition,
            coord_dim=coord_dim
        )
        self.coord_dim=coord_dim

    def query_opacity(self, x, step_size):
        density = self.query_density(x)
        # if the density is small enough those two are the same.
        # opacity = 1.0 - torch.exp(-density * step_size)
        opacity = density * step_size
        return opacity

    def query_density(self, x, coord=None):
        if self.coord_dim==3:
            coord = x.clone()*2.0/3.0
        x = self.posi_encoder(x)
        sigma = self.mlp.query_density(x, coord=coord)
        return F.relu(sigma)

    def forward(self, x, condition=None, coord=None):
        if self.coord_dim==3:
            coord = x.clone()*2.0/3.0
        x = self.posi_encoder(x)
        if condition is not None:
            condition = self.view_encoder(condition)
        rgb, sigma = self.mlp(x, condition=condition, coord=coord)
        return torch.sigmoid(rgb), F.relu(sigma)


class TNeRFRadianceField(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.posi_encoder = SinusoidalEncoder(3, 0, 4, True)
        self.time_encoder = SinusoidalEncoder(1, 0, 4, True)
        self.warp = MLP(
            input_dim=self.posi_encoder.latent_dim
            + self.time_encoder.latent_dim,
            output_dim=3,
            net_depth=4,
            net_width=64,
            skip_layer=2,
            output_init=functools.partial(torch.nn.init.uniform_, b=1e-4),
        )
        self.nerf = VanillaNeRFRadianceField(coord_dim=1)
        
    def query_opacity(self, x, timestamps, step_size):

        idxs = torch.randint(0, len(timestamps), (x.shape[0],), device=x.device)
        t = timestamps[idxs]
        density = self.query_density(x, t)
        # if the density is small enough those two are the same.
        # opacity = 1.0 - torch.exp(-density * step_size)
        opacity = density * step_size
        return opacity

    def query_density(self, x, t):
        coord = 2*t-1.0
        x = x + self.warp(
            torch.cat([self.posi_encoder(x), self.time_encoder(t)], dim=-1), coord=None
        )
        return self.nerf.query_density(x, coord=coord)

    def forward(self, x, t, condition=None):
        coord = 2*t-1.0
        x = x + self.warp(
            torch.cat([self.posi_encoder(x), self.time_encoder(t)], dim=-1), coord=None
        )
        return self.nerf(x, condition=condition, coord=coord)

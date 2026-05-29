# FreeTimeGS++
# 2025-2026 Lucas Yunkyu Lee <lucaslee@postech.ac.kr>, SNU VGI Lab

from typing import Callable, Sequence, TypeAlias

import tinycudann as tcnn
import torch
from torch import Tensor, nn

ExplicitVelocities: TypeAlias = nn.Parameter


class Bounds(nn.Module):
    """N-Dimensional bounds, used for normalizing coordinates"""

    _bounds: nn.Parameter  # (dim, 2) in min, max order

    def __init__(self, bounds: Sequence[float] | Sequence[Sequence[float]] | Tensor):
        super().__init__()
        if not isinstance(bounds, Tensor):
            bounds = torch.tensor(bounds)
        assert bounds.shape[-1] == 2 and bounds.dim() <= 2
        self._bounds = nn.Parameter(bounds.reshape(-1, 2), False)

    def dim(self) -> int:
        return len(self._bounds)

    def normalize(
        self,
        input: Tensor,  # (..., dim)
        range: tuple[float, float] = (0, 1),
    ):
        assert input.shape[-1] == self.dim()
        min, max = range

        scale = self._bounds[:, 1] - self._bounds[:, 0]
        shift = self._bounds[:, 0]
        z = (input - shift) / scale

        return (max - min) * z + min


class MLP(nn.Module):
    """Basic MLP"""

    layers: nn.ModuleList
    activation: Callable[[Tensor], Tensor]

    def __init__(
        self,
        *,
        input_dims: int,
        output_dims: int,
        hidden_dims: int,
        hidden_layers: int,
        activation: Callable[[Tensor], Tensor],
    ):
        super().__init__()

        layers = [
            nn.Linear(input_dims, hidden_dims),
            *[nn.Linear(hidden_dims, hidden_dims) for _ in range(hidden_layers - 1)],
            nn.Linear(hidden_dims, output_dims),
        ]
        self.layers = nn.ModuleList(layers)
        self.activation = activation

    def forward(self, input: Tensor):
        z = input
        for linear in self.layers[:-1]:
            z = linear(z)
            z = self.activation(z)
        z = self.layers[-1](z)

        return z


class HyperMLP(nn.Module):
    """MLP with biases modulated with hypernetworks"""

    layers: nn.ModuleList
    bias_nets: nn.ModuleList

    activation: Callable[[Tensor], Tensor]
    hidden_dims: int
    hidden_layers: int

    def __init__(
        self,
        *,
        input_dims: int,
        output_dims: int,
        hidden_dims: int,
        hidden_layers: int,
        activation: Callable[[Tensor], Tensor],
        bias_net_input_dims: int,
        bias_net_hidden_dims: int,
        bias_net_hidden_layers: int,
        bias_net_activation: Callable[[Tensor], Tensor],
    ):
        super().__init__()

        layers = [
            nn.Linear(input_dims, hidden_dims, bias=False),
            *[
                nn.Linear(hidden_dims, hidden_dims, bias=False)
                for _ in range(hidden_layers - 1)
            ],
            nn.Linear(hidden_dims, output_dims, bias=False),
        ]
        self.layers = nn.ModuleList(layers)
        self.activation = activation

        bias_nets = [
            MLP(
                input_dims=bias_net_input_dims,
                output_dims=hidden_dims,
                hidden_dims=bias_net_hidden_dims,
                hidden_layers=bias_net_hidden_layers,
                activation=bias_net_activation,
            )
            for _ in range(hidden_layers)
        ]
        bias_nets.append(
            MLP(
                input_dims=bias_net_input_dims,
                output_dims=output_dims,
                hidden_dims=bias_net_hidden_dims,
                hidden_layers=bias_net_hidden_layers,
                activation=bias_net_activation,
            )
        )
        self.bias_nets = nn.ModuleList(bias_nets)

        self.hidden_dims = hidden_dims
        self.hidden_layers = hidden_layers

    def forward(self, input: Tensor, z_bias: Tensor):
        z = input.float()
        for i, linear in enumerate(self.layers[:-1]):
            z = linear(z) + self.bias_nets[i](z_bias)
            z = self.activation(z)

        z = self.layers[-1](z) + self.bias_nets[-1](z_bias)

        return z


class VelocityField(nn.Module):
    enc_xyz: tcnn.Encoding
    enc_t: tcnn.Encoding
    hyper_mlp: nn.Module

    bounds_xyz: Bounds
    bounds_t: Bounds

    @torch.no_grad()
    def __init__(self, bounds_xyz: Bounds, bounds_t: Bounds):
        super().__init__()

        self.bounds_xyz = bounds_xyz
        self.bounds_t = bounds_t

        self.enc_xyz = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "Grid",
                "type": "Hash",
                "n_levels": 16,
                "n_features_per_level": 2,
                "log2_hashmap_size": 19,
                "base_resolution": 16,
                "per_level_scale": 2,
                "interpolation": "Linear",
            },
        )
        self.enc_t = tcnn.Encoding(
            n_input_dims=1,
            encoding_config={"otype": "Identity"},
        )
        self.hyper_mlp = HyperMLP(
            input_dims=self.enc_xyz.n_output_dims,
            output_dims=3,
            hidden_dims=128,
            hidden_layers=3,
            activation=nn.functional.relu,
            bias_net_input_dims=self.enc_t.n_output_dims,
            bias_net_hidden_dims=128,
            bias_net_hidden_layers=3,
            bias_net_activation=nn.functional.relu,
        )
        self.hyper_mlp.compile()

    def forward(self, xyz: Tensor, t: float | Tensor):
        if not isinstance(t, Tensor):
            t = torch.tensor(t, device="cuda")
        t = t.reshape(-1, 1)

        xyz = self.bounds_xyz.normalize(xyz, (0, 1))
        t = self.bounds_t.normalize(t, (0, 1))
        z_xyz = self.enc_xyz(xyz).float()
        z_t = self.enc_t(t).float()

        v = self.hyper_mlp(z_xyz, z_t)

        return v

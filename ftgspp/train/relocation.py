# FreeTimeGS++
# 2025-2026 Lucas Yunkyu Lee <lucaslee@postech.ac.kr>, SNU VGI Lab

import math
from importlib.metadata import PackageNotFoundError, version

import torch
from gsplat.relocation import compute_relocation
from gsplat.strategy.ops import _multinomial_sample, _update_param_with_optimizer
from torch import Tensor, nn

from ftgspp.models.gaussians import Gaussians
from ftgspp.train.optim import OptimizerCollection

REQUIRED_GSPLAT_VERSION = "1.5.3"

try:
    INSTALLED_GSPLAT_VERSION = version("gsplat")
except PackageNotFoundError as exc:
    raise RuntimeError(
        "ftgspp.train.relocation requires gsplat to be installed separately. "
        f"Install gsplat=={REQUIRED_GSPLAT_VERSION}."
    ) from exc

if INSTALLED_GSPLAT_VERSION != REQUIRED_GSPLAT_VERSION:
    raise RuntimeError(
        "ftgspp.train.relocation is validated only with "
        f"gsplat=={REQUIRED_GSPLAT_VERSION}, but found "
        f"gsplat=={INSTALLED_GSPLAT_VERSION}."
    )


def relocation_binoms() -> Tensor:
    n_max = 51
    binoms = torch.zeros((n_max, n_max))
    for n in range(n_max):
        for k in range(n + 1):
            binoms[n, k] = math.comb(n, k)
    return binoms


@torch.no_grad()
def relocate(
    gs: Gaussians,
    optimizers: OptimizerCollection,
    probs: Tensor,
    state: dict[str, Tensor],
    mask: Tensor,
    binoms: Tensor,
    min_opacity: float = 0.005,
    mode: str = "3d_mcmc",
):
    """Thin wrapper around gsplat relocation with FTGS++ sampling scores."""
    if mode != "3d_mcmc":
        raise ValueError(
            f"Unsupported relocation mode: {mode}. "
            "This release only supports '3d_mcmc'."
        )

    dead_indices = mask.nonzero(as_tuple=True)[0]
    alive_indices = (~mask).nonzero(as_tuple=True)[0]
    n_dead = int(dead_indices.numel())
    if n_dead == 0:
        return

    sampled_local = _multinomial_sample(
        probs[alive_indices].flatten(),
        n_dead,
        replacement=True,
    )
    source_indices = alive_indices[sampled_local]
    _relocate_mcmc(
        gs=gs,
        optimizers=optimizers,
        state=state,
        dead_indices=dead_indices,
        source_indices=source_indices,
        binoms=binoms,
        min_opacity=min_opacity,
    )


@torch.no_grad()
def _relocate_mcmc(
    gs: Gaussians,
    optimizers: OptimizerCollection,
    state: dict[str, Tensor],
    dead_indices: Tensor,
    source_indices: Tensor,
    binoms: Tensor,
    min_opacity: float,
):
    eps = torch.finfo(torch.float32).eps
    opacities = torch.sigmoid(gs.opacities)
    new_opacities, new_scales = compute_relocation(
        opacities=opacities[source_indices],
        scales=torch.exp(gs.scales)[source_indices],
        ratios=torch.bincount(source_indices)[source_indices] + 1,
        binoms=binoms,
    )
    new_opacities = torch.clamp(new_opacities, max=1.0 - eps, min=min_opacity)

    def param_fn(name: str, param: Tensor) -> nn.Parameter:
        updated = param.detach().clone()
        if name == "opacities":
            updated[source_indices] = torch.logit(new_opacities)
        elif name == "scales":
            updated[source_indices] = torch.log(new_scales)
        updated[dead_indices] = updated[source_indices]
        return nn.Parameter(updated, requires_grad=param.requires_grad)

    def optimizer_fn(_: str, value: Tensor) -> Tensor:
        updated = value.clone()
        updated[source_indices] = 0
        return updated

    _update_param_with_optimizer(
        param_fn=param_fn,
        optimizer_fn=optimizer_fn,
        # Update the module's direct parameter registry in-place.
        params=gs._parameters,
        optimizers=optimizers.asdict(),
    )

    for key, value in state.items():
        if isinstance(value, Tensor):
            value[source_indices] = 0

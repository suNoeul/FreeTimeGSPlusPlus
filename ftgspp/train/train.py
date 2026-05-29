# FreeTimeGS++
# 2025-2026 Lucas Yunkyu Lee <lucaslee@postech.ac.kr>, SNU VGI Lab

import logging
import random
from dataclasses import asdict
from pathlib import Path
from typing import Mapping, Sequence

import rerun as rr
import torch
from fused_ssim import fused_ssim
from torch import Tensor, nn
from torchmetrics.image import (
    LearnedPerceptualImagePatchSimilarity,
    PeakSignalNoiseRatio,
)
from tqdm import tqdm

from ftgspp.data.utils import FastInfiniteLoader, MultiViewData
from ftgspp.eval import eval_rgb
from ftgspp.models import ColorCorrectors, Gaussians, VelocityField
from ftgspp.train.optim import (
    LRSchedulerCollection,
    OptimizerCollection,
)
from ftgspp.train.relocation import relocate, relocation_binoms
from ftgspp.train.state import TrainState
from ftgspp.utils import tensor_to_png
from ftgspp.utils.math import chw, scene_extent


def train(
    gs: Gaussians,
    color_correctors: ColorCorrectors | None,
    dataset: MultiViewData,
    batch_size: int,
    iterations: int,
    frames: slice,
    train_cameras: slice | Sequence[int],
    eval_cameras: slice | Sequence[int],
    relocation: bool,
    relocation_start: int,
    relocation_every: int,
    relocation_stop: int,
    relocation_opacity_threshold: float,
    relocation_mode: str,
    relocation_score_mode: str,
    relocation_score_mode_start: int,
    color_correction_start: int,
    run_path: Path,
    loss_weights: Mapping[str, float],
    sh_degree_schedule: bool,
    lrs: Mapping[str, float],
    lr_schedules: Mapping[str, float],
    checkpoint_iterations: list[int],
    logger: logging.Logger,
    start_iteration: int = 0,
    velocity_distill: tuple[Tensor, Tensor, Tensor] | None = None,
    velocity_distill_warmup_steps: int = 0,
    velocity_distill_weight: float = 0.0,
    velocity_distill_batch_size: int = 32768,
):
    print(f"Using {len(gs)} Gaussians")

    # Data
    train_set = dataset.at(frames, train_cameras)
    loader = FastInfiniteLoader[MultiViewData](train_set, None)  # type: ignore

    # Optimizers
    optimizers, schedulers = make_optimizers(
        gs=gs,
        color_correctors=color_correctors,
        lrs=lrs,
        lr_schedules=lr_schedules,
        batch_size=batch_size,
        iterations=iterations,
        scene_extent=scene_extent(train_set[0].w2c),  # type: ignore
    )

    # Relocation
    state = None
    binoms = None
    if relocation:
        state = TrainState.new(len(gs))
        binoms = relocation_binoms().cuda()

    psnr_fn = PeakSignalNoiseRatio(data_range=1).cuda()
    lpips_fn = None
    if loss_weights["lpips"] > 0:
        lpips_fn = LearnedPerceptualImagePatchSimilarity(
            net_type="alex", normalize=True
        ).cuda()
    velocity_distill_xyz = None
    velocity_distill_t = None
    velocity_distill_target = None
    velocity_distill_scale = None
    if velocity_distill is not None:
        velocity_distill_xyz, velocity_distill_t, velocity_distill_target = velocity_distill
        velocity_distill_xyz = velocity_distill_xyz.to(gs.means.device)
        velocity_distill_t = velocity_distill_t.to(gs.means.device)
        velocity_distill_target = velocity_distill_target.to(gs.means.device)
        velocity_distill_scale = torch.linalg.norm(
            velocity_distill_target, dim=-1
        ).mean().clamp(min=1e-6)

    postfix = {}

    for it in (bar := tqdm(range(start_iteration, start_iteration + iterations))):
        rr.set_time("iteration", sequence=it)

        batch: MultiViewData = next(loader)
        batch_idxs = random.choices(range(len(batch)), k=batch_size)
        batch = batch[batch_idxs].cuda()  # type: ignore

        t = batch.time[0]
        cams: list[int] = batch.camera  # type: ignore

        pred_, _, aux = gs(
            t=t,
            w2c=batch.w2c,
            intrinsic=batch.intrinsic,
            shape=(batch.height, batch.width),
            sh_degree=min(it // 1000, gs.sh_degree)
            if sh_degree_schedule
            else gs.sh_degree,
        )
        if color_correctors is not None and it > color_correction_start:
            cc = color_correctors
            pred = cc(cams, pred_)
        else:
            pred = pred_
        aux["means2d"].retain_grad()

        gt = batch.rgb.float() / 255
        zero = torch.zeros((), device=pred.device)
        loss_l1 = loss_weights["l1"] * nn.functional.l1_loss(pred, gt)
        loss_ssim = loss_weights["ssim"] * (
            1 - fused_ssim(chw(pred), chw(gt), padding="valid")
        )
        if lpips_fn is not None:
            # Color correction can push RGB slightly outside [0, 1].
            # LPIPS in torchmetrics validates strict input bounds.
            pred_lpips = pred.clamp(0, 1)
            gt_lpips = gt.clamp(0, 1)
            loss_lpips = loss_weights["lpips"] * lpips_fn(chw(pred_lpips), chw(gt_lpips))
        else:
            loss_lpips = zero
        if lpips_fn is not None:
            lpips_fn.reset()
        loss_reg_opacity = loss_weights["reg_opacity"] * torch.mean(
            gs.opacities.sigmoid() * gs.temporal_opacity(t).detach()
        )
        loss_reg_scale = loss_weights["reg_scale"] * gs.scales.exp().mean()
        loss_reg = loss_reg_opacity + loss_reg_scale
        loss_gate = loss_weights["gate"] * torch.mean(1 - gs.gate())
        loss_color_correctors = (
            loss_weights["color_correctors"] * color_correctors.regularize()
            if color_correctors is not None
            else zero
        )
        loss_velocity_distill = zero
        if (
            velocity_distill_xyz is not None
            and velocity_distill_t is not None
            and velocity_distill_target is not None
            and velocity_distill_scale is not None
            and velocity_distill_weight > 0
            and velocity_distill_warmup_steps > 0
            and (it - start_iteration) < velocity_distill_warmup_steps
            and isinstance(gs.velocity_model, VelocityField)
        ):
            n_total = len(velocity_distill_target)
            n_sample = min(velocity_distill_batch_size, n_total)
            sample_idxs = torch.randint(
                0,
                n_total,
                (n_sample,),
                device=velocity_distill_target.device,
            )
            pred_velocity = gs.velocity_model(
                velocity_distill_xyz[sample_idxs],
                velocity_distill_t[sample_idxs],
            )
            target_velocity = velocity_distill_target[sample_idxs]
            warmup_ratio = 1 - (it - start_iteration) / velocity_distill_warmup_steps
            loss_velocity_distill = (
                velocity_distill_weight
                * warmup_ratio
                * nn.functional.smooth_l1_loss(
                    pred_velocity / velocity_distill_scale,
                    target_velocity / velocity_distill_scale,
                )
            )

        loss = (
            loss_l1
            + loss_ssim
            + loss_lpips
            + loss_reg
            + loss_gate
            + loss_color_correctors
            + loss_velocity_distill
        )

        loss.backward()

        optimizers.step()
        schedulers.step()
        optimizers.zero_grad()

        if relocation:
            assert state is not None
            assert binoms is not None
            with torch.no_grad():
                state.update_(aux)

                if (
                    it % relocation_every == 0
                    and relocation_start < it < relocation_stop
                ):
                    relocated, is_all_dead = relocate_gaussians(
                        state=state,
                        gs=gs,
                        opacity_threshold=relocation_opacity_threshold,
                        mode=relocation_mode,
                        score_mode=relocation_score_mode,
                        score_mode_start=relocation_score_mode_start,
                        optimizers=optimizers,
                        binoms=binoms,
                        it=it,
                        logger=logger,
                    )
                    state.zero_()
                    postfix.update({"relocated": f"{relocated}"})
                    if is_all_dead:
                        logger.warning(
                            "Stopping early at iteration %d: all Gaussians are dead "
                            "(threshold=%.6f).",
                            it,
                            relocation_opacity_threshold,
                        )
                        bar.set_postfix(postfix)
                        return gs

        if it in checkpoint_iterations:
            ckpt_dir = run_path / "ckpt"
            ckpt_dir.mkdir(exist_ok=True)
            gs.save(ckpt_dir / f"{it:06d}.pt")
            with torch.inference_mode():
                psnr = eval_rgb(
                    gs=gs,
                    eval_set=dataset.at(frames, eval_cameras),
                    metric_fns={"psnr": psnr_fn},
                ).mean()["psnr"]
                logger.info(f"Checkpoint {it:05d} PSNR {psnr:.2f}")

        if it % 500 == 0:
            with torch.inference_mode():
                psnr = eval_rgb(
                    gs=gs,
                    eval_set=dataset.at(frames, eval_cameras),
                    metric_fns={"psnr": psnr_fn},
                ).mean()["psnr"]
                postfix.update({"psnr": f"{psnr:.2f}"})
                rr.log("psnr", rr.Scalars(psnr))
                logger.info(f"Iteration {it:05d} PSNR {psnr:.2f}")

            rr.log(
                "gt",
                rr.EncodedImage(contents=tensor_to_png(gt[0]), media_type="image/png"),
            )
            rr.log(
                "pred",
                rr.EncodedImage(
                    contents=tensor_to_png(pred[0]), media_type="image/png"
                ),
            )
            rr.log(
                "pred-raw",
                rr.EncodedImage(
                    contents=tensor_to_png(pred_[0]), media_type="image/png"
                ),
            )

        rr.log("loss/total", rr.Scalars(float(loss.detach())))
        rr.log("loss/l1", rr.Scalars(float(loss_l1.detach())))
        rr.log("loss/ssim", rr.Scalars(float(loss_ssim.detach())))
        rr.log("loss/lpips", rr.Scalars(float(loss_lpips.detach())))
        rr.log("loss/reg", rr.Scalars(float(loss_reg.detach())))
        rr.log("loss/gate", rr.Scalars(float(loss_gate.detach())))
        rr.log(
            "loss/color_correctors", rr.Scalars(float(loss_color_correctors.detach()))
        )
        rr.log("loss/velocity_distill", rr.Scalars(float(loss_velocity_distill.detach())))

        bar.set_postfix(postfix)

    return gs


def relocate_gaussians(
    state: TrainState,
    gs: Gaussians,
    opacity_threshold: float,
    mode: str,
    score_mode: str,
    score_mode_start: int,
    optimizers: OptimizerCollection,
    binoms: Tensor,
    it: int,
    logger: logging.Logger,
) -> tuple[int, bool]:
    opacities = torch.sigmoid(gs.opacities).flatten()
    opacities_probs = opacities / opacities.sum()
    grad2d = state.grad2d_acc()
    grad2d_probs = grad2d / grad2d.sum()
    probs = opacities_probs + grad2d_probs

    if score_mode == "gate_included":
        gate_scores = (1 - gs.gate().flatten())
        if score_mode_start <= it:
            gate_probs = gate_scores / gate_scores.sum()
            probs += gate_probs
    if (not torch.isfinite(probs).all()):
        probs = torch.ones_like(probs)

    dead_mask = opacities <= opacity_threshold
    n_gs = int(dead_mask.sum().item())
    n_alive = int((~dead_mask).sum().item())
    if n_gs > 0 and n_alive == 0:
        logger.warning(
            "Relocation skipped: alive Gaussian count is zero (dead=%d, total=%d).",
            n_gs,
            len(gs),
        )
        return n_gs, True
    if n_gs > 0:
        logger.debug(
            "Relocation trigger mode=%s score_mode=%s dead=%d",
            mode,
            score_mode,
            n_gs,
        )
        relocate(
            gs,
            optimizers=optimizers,
            probs=probs,
            state=asdict(state),
            mask=dead_mask,
            binoms=binoms,
            min_opacity=opacity_threshold,
            mode=mode,
        )
        logger.debug("Relocation applied relocated=%d", n_gs)

    return n_gs, False


def make_optimizers(
    gs: Gaussians,
    color_correctors: ColorCorrectors | None,
    *,
    lrs: Mapping[str, float],
    lr_schedules: Mapping[str, float],
    batch_size: int,
    iterations: int,
    scene_extent: float,
) -> tuple[OptimizerCollection, LRSchedulerCollection]:
    optimizers = {
        name: torch.optim.Adam(
            [{"params": param, "lr": lrs[name] * batch_size**0.5, "name": name}],
            eps=1e-15 / batch_size**0.5,
            betas=(0.9**batch_size, 0.999**batch_size),
        )
        for name, param in gs.named_parameters(recurse=False)
        if name in lrs and param.requires_grad
    }
    for name, module in gs.named_modules():
        if name in lrs and any(p.requires_grad for p in module.parameters()):
            optimizers[name] = torch.optim.Adam(
                [
                    {
                        "params": module.parameters(),
                        "lr": lrs[name] * batch_size**0.5,
                        "name": name,
                    }
                ],
                eps=1e-15 / batch_size**0.5,
                betas=(0.9**batch_size, 0.999**batch_size),
            )

    if "color_correctors" in lrs and color_correctors is not None:
        optimizers["color_correctors"] = torch.optim.Adam(
            [
                {
                    "params": color_correctors.parameters(),
                    "lr": lrs["color_correctors"] * batch_size**0.5,
                    "name": "color_correctors",
                }
            ],
            eps=1e-15 / batch_size**0.5,
            betas=(0.9**batch_size, 0.999**batch_size),
        )

    if "means" in optimizers:
        optimizers["means"].param_groups[0]["lr"] *= scene_extent
    if "velocity_model" in optimizers:
        optimizers["velocity_model"].param_groups[0]["lr"] *= scene_extent

    lr_schedulers: dict[str, torch.optim.lr_scheduler.LRScheduler] = {
        name: torch.optim.lr_scheduler.ExponentialLR(
            optimizers[name],
            schedule ** (1 / iterations),
        )
        for name, schedule in lr_schedules.items()
        if name in optimizers
    }

    optimizers = OptimizerCollection(optimizers)
    lr_schedulers = LRSchedulerCollection(lr_schedulers)

    return optimizers, lr_schedulers

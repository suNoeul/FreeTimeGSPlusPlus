<div align="center">
  <h1>FreeTimeGS++: Secrets of Dynamic Gaussian<br />Splatting and Their Principles</h1>
  <a href="https://arxiv.org/pdf/2605.03337">
    <img alt="Paper" src="https://img.shields.io/badge/paper-blue" />
  </a>
  <a href="https://arxiv.org/abs/2605.03337">
    <img alt="arXiv" src="https://img.shields.io/badge/arXiv-2605.03337-b31b1b" />
  </a>
  <a href="https://yklcs.com/ftgspp">
    <img alt="Project Page" src="https://img.shields.io/badge/Project_Page-green" />
  </a>
  <a href="#citation">
    <img alt="BibTeX" src="https://img.shields.io/badge/Citation-BibTeX-8250df" />
  </a>
  <br />
  <br />
  <strong><sup>1</sup> Seoul National University &nbsp; <sup>2</sup> POSTECH</strong>
  <br />
  <br />
  <a href="https://yklcs.com/">Lucas Yunkyu Lee</a><sup>1,2,*</sup>,
  <a href="https://sunoeul.github.io/">Soonho Kim</a><sup>1,*</sup>,
  <a href="https://ywk02.github.io/">Youngwook Kim</a><sup>1</sup>,
  <a href="https://nstar1125.github.io/">Sangmin Kim</a><sup>1</sup>,
  <a href="https://jaesik.info/">Jaesik Park</a><sup>1,&dagger;</sup>
  <br />
  <br />
  * Equal contribution. <sup>&dagger;</sup> Corresponding author.
</div>

## License

This repository is released under the non-commercial research and educational use license in [LICENSE](LICENSE). It is a source-available research release and is not distributed under an OSI-approved open-source license.

The FreeTimeGS++ research license applies only to the original code in this repository that is owned by the authors. Third-party software, models, weights, and datasets remain under their own license terms. In particular, this project depends on external packages such as `gsplat` (Apache-2.0), `pycolmap` (BSD-3-Clause), PyTorch/torchvision (BSD-style), RoMa (MIT), MEMFOF (BSD-3-Clause), `fused-ssim` (MIT), `tiny-cuda-nn` (BSD-3-Clause), and `rerun-sdk` (MIT OR Apache-2.0). See [THIRD_PARTY.md](THIRD_PARTY.md) for a concise inventory.

## Requirements

FreeTimeGS++ currently targets Python 3.12 and CUDA 12.8. The repository also assumes standard system tools such as `ffmpeg` and native build tooling for Python extensions. See [Dockerfile](Dockerfile), [flake.nix](flake.nix), and [uv.lock](uv.lock) for the reference system and Python environment definitions.

## Installation

[uv](https://github.com/astral-sh/uv) is the recommended environment manager.

This repository intentionally does not declare `gsplat` as an automatic Python dependency. If you want to run the training, initialization, or rendering pipeline, you must obtain and install `gsplat` separately from its upstream project under the Apache-2.0 license.

This release is validated against `gsplat==1.5.3`. In particular, the relocation path uses `gsplat` private strategy helpers, so using a newer or older version is not recommended unless you re-validate compatibility yourself.

```shell
git clone https://github.com/yklcs/FreeTimeGSPlusPlus.git
cd FreeTimeGSPlusPlus

uv sync --group with-torch
uv pip install "gsplat==1.5.3"
```

If you need a different installation method, follow the upstream instructions at <https://github.com/nerfstudio-project/gsplat>, but keep the version pinned to `1.5.3` for this release unless you have re-validated the code against another version. Most core commands in this repository require `gsplat` to be present in the environment.

## Data Setup

The released configs assume that the input datasets are available under `/data/dynerf` and `/data/selfcap`. The DyNeRF scenes can be obtained from [Neural 3D Video](https://github.com/facebookresearch/Neural_3D_Video), and the SelfCap scenes can be obtained from [SelfCap-Dataset](https://huggingface.co/datasets/zju3dv/SelfCap-Dataset). We recommend keeping the downloaded datasets read-only and mapping them into that layout with symbolic links, so that the released configs can be used without editing input paths. In the released configs, the other paths such as `_extracted`, `_colmap`, `_memmap`, and `_points` are generated artifacts and can remain unchanged.

```shell
ln -s /path/to/dynerf_root /data/dynerf
ln -s /path/to/selfcap_root /data/selfcap
```

The expected directory layout is:

- `/data/dynerf/<scene>/...`
- `/data/selfcap/{bike,corgi,hair}/...`

## Main Usage

The main entry point of this repository is [`./run`](./run).

```shell
./run {dataset_name} {configs_path} {output_path} [options]
```

`dataset_name` currently supports `dynerf` and `selfcap`.

For each dataset, the released configs are grouped as:

- `configs/<dataset>/ftgs`: FreeTimeGS baseline
- `configs/<dataset>/ftgspp`: final FreeTimeGS++ preset
- `configs/<dataset>/ablation/...`: additional analysis presets

The simplest way to test the full pipeline is:

```shell
./run dynerf configs/dynerf/ftgs _run/dynerf --scenes coffee_martini
```

To run the final FreeTimeGS++ preset instead of the baseline, use:

```shell
./run dynerf configs/dynerf/ftgspp _run/ftgspp_dynerf --scenes coffee_martini
```

This runs the default pipeline from `extract` to `render`. The script also supports partial execution with `--from` and `--to`, repeated runs with `-n`, and statistics-only aggregation with `--stats`.

```shell
./run dynerf configs/dynerf/ftgs _run/dynerf --scenes coffee_martini --from train --to render
./run dynerf configs/dynerf/ftgs _run/dynerf_repeat --scenes coffee_martini -n 3
./run dynerf configs/dynerf/ftgs _run/dynerf_repeat --scenes coffee_martini -n 3 --stats
```

The explicit module-by-module commands used in `scripts/run_dynerf` and `scripts/run_selfcap` remain available when step-by-step execution or debugging is needed.

## UFM Precomputation

If you want to use the UFM-based initialization variant (`init.temporal_motion_adapted = true`), dense correspondence flow from [UFM](https://uniflowmatch.github.io/) must be precomputed before `ftgspp.init`. For a single scene, this can be done with:

```shell
uv run -m ftgspp.data.flow configs/dynerf/ftgspp/flame_steak.toml
```

Since this preprocessing can be time-consuming, the repository also provides [`scripts/run_ufm_flow`](scripts/run_ufm_flow) to precompute and cache UFM flow for the released scenes in advance.

`ftgspp.data.flow` expects an external UFM installation (`ufm` or `uniflowmatch`) and is not installed by default through this repository's Python dependencies.

## Citation

```bibtex
@misc{ftgspp,
      title={FreeTimeGS++: Secrets of Dynamic Gaussian Splatting and Their Principles}, 
      author={Lucas Yunkyu Lee and Soonho Kim and Youngwook Kim and Sangmin Kim and Jaesik Park},
      year={2026},
      eprint={2605.03337},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2605.03337}, 
}
```

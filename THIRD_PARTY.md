# Third-Party Software

This source repository depends on third-party software, models, weights, and
datasets that are not covered by the FreeTimeGS++ research license in
[LICENSE](LICENSE). Each dependency remains subject to its own upstream license
terms.

The lists below cover selected major direct dependencies and optional external
assets referenced by this repository. They are not a substitute for reviewing
each upstream project and its transitive dependencies before redistribution.

This repository does not vendor the listed third-party packages, model weights,
datasets, or container images.

## Software

| Component | Upstream | License |
| --- | --- | --- |
| `gsplat` | https://github.com/nerfstudio-project/gsplat | Apache-2.0 |
| `pycolmap` / COLMAP | https://github.com/colmap/colmap | BSD-3-Clause |
| `torch` | https://github.com/pytorch/pytorch | BSD-style / BSD-3-Clause |
| `torchvision` | https://github.com/pytorch/vision | BSD-style |
| `romatch` (RoMa) | https://github.com/Parskatt/RoMa | MIT |
| `romav2` | https://github.com/Parskatt/RoMaV2 | MIT |
| `fused-ssim` | https://github.com/MrNeRF/optimized-fused-ssim | MIT |
| `tinycudann` | https://github.com/NVlabs/tiny-cuda-nn | BSD-3-Clause |
| `memfof` | https://github.com/msu-video-group/memfof | BSD-3-Clause |
| `rerun-sdk` | https://github.com/rerun-io/rerun | MIT OR Apache-2.0 |
| `kornia` | https://github.com/kornia/kornia | Apache-2.0 |
| `cuml-cu12` | https://github.com/rapidsai/cuml | Apache-2.0 |
| `flip-evaluator` | https://github.com/NVlabs/flip | BSD-3-Clause |

## Models and Weights

| Component | Upstream | License |
| --- | --- | --- |
| MEMFOF pretrained weights | https://huggingface.co/egorchistov/optical-flow-MEMFOF-Tartan-T-TSKH | BSD-3-Clause |
| UFM pretrained weights | https://huggingface.co/infinity1096/UFM-Base | CC-BY-NC-4.0 |

## Datasets

| Component | Upstream |
| --- | --- |
| Neural 3D Video / DyNeRF scenes | https://github.com/facebookresearch/Neural_3D_Video |
| SelfCap Dataset | https://huggingface.co/datasets/zju3dv/SelfCap-Dataset |

Notes:

- `gsplat` is intentionally not declared as an automatic dependency in this
  repository's package metadata. Users are expected to obtain and install
  `gsplat==1.5.3` separately from its upstream project under Apache-2.0.
- This release is validated against `gsplat==1.5.3`. The relocation path uses
  `gsplat` private strategy helpers from that version.
- Most other Python packages listed in `pyproject.toml` may still be installed
  through the repository's environment setup commands.
- `tinycudann` is installed from a repository fork configured in
  `pyproject.toml`; its upstream project is NVIDIA's `tiny-cuda-nn`.
- UFM is an optional external installation used only for the UFM-based
  initialization variant.
- If you redistribute binaries, containers, or packaged environments built
  from this repository, you are responsible for satisfying the notice and
  license obligations of the bundled third-party software.

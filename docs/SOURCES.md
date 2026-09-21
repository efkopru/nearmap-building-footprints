# Primary sources

Checked **2026-09-20**. These sources establish upstream capabilities and setup requirements. They do not establish this project's execution status or accuracy. Links to `main` and `latest` can change; record the exact source revision and local model hashes for each actual run. Sampling, comparison groups, and reporting rules in [METHODOLOGY.md](METHODOLOGY.md) are this project's proposed evaluation protocol, not vendor performance guarantees.

Upstream SAM 3 `main` was resolved during implementation to commit [`2345a4ad109ac29c569da749c91d84f10dc08c40`](https://github.com/facebookresearch/sam3/tree/2345a4ad109ac29c569da749c91d84f10dc08c40). Preserve the revision actually installed for a run rather than assuming a later `main` still has identical interfaces. The implementation pins the SamGeo wrapper separately; native Meta capabilities do not automatically imply identical wrapper behavior.

| Source | Exact URL | Used for |
| --- | --- | --- |
| Meta SAM 3 repository | https://github.com/facebookresearch/sam3 | Concept segmentation, model access, Python/PyTorch/CUDA prerequisites. Pin the SAM 3 image checkpoint and compatible code; a newer video release is not automatically an equivalent image baseline. |
| Meta image prompting example | https://github.com/facebookresearch/sam3/blob/main/examples/sam3_image_predictor_example.ipynb | Text and positive/negative visual box examples. |
| Meta image processor | https://github.com/facebookresearch/sam3/blob/main/sam3/model/sam3_image_processor.py | `set_text_prompt`; `add_geometric_prompt`; normalized center-format concept boxes and Boolean labels; prompt state. |
| Meta image model | https://github.com/facebookresearch/sam3/blob/main/sam3/model/sam3_image.py | Separate `predict_inst` path for instance interaction. |
| Meta instance predictor | https://github.com/facebookresearch/sam3/blob/main/sam3/model/sam1_task_predictor.py | Per-instance box/point inputs, coordinate conventions, foreground/background labels, and predicted mask quality. |
| SamGeo 3 wrapper reference | https://samgeo.gishub.org/samgeo3/ | Geospatial wrapper methods and their implementation. Record the installed package revision; its convenience methods have distinct state behavior. |
| SamGeo 1.4.2 distribution | https://pypi.org/project/segment-geospatial/1.4.2/ | Version-specific package provenance for the adapter; separate from the native Meta model repository. |
| Meta training guide | https://github.com/facebookresearch/sam3/blob/main/README_TRAIN.md | Training extras, local and distributed training, Hydra configuration, checkpoint and log outputs. Follow the pinned version's actual dataset contract and entry point. |
| Meta evaluation/trainer configuration | https://github.com/facebookresearch/sam3/blob/main/sam3/train/configs/eval_base.yaml | Upstream CUDA accelerator and NCCL backend configuration. |
| Triton supported platforms | https://github.com/triton-lang/triton#compatibility | Linux platform support and hardware compatibility; supports selecting a Linux training environment. |
| NVIDIA CUDA on WSL | https://docs.nvidia.com/cuda/wsl-user-guide/index.html | Windows-host driver and Linux CUDA workflows under WSL2; do not install a Linux display driver inside WSL. This is NVIDIA platform guidance, not a Meta certification of this project. |
| Esri Building Footprint Extraction USA | https://doc.arcgis.com/en/pretrained-models/latest/imagery/introduction-to-building-footprint-extraction-usa.htm | Mask R-CNN architecture and model scope. Vendor benchmark figures are not local results. |
| Esri Detect Objects Using Deep Learning | https://doc.esri.com/en/arcgis-pro/latest/tool-reference/image-analyst/detect-objects-using-deep-learning.html | Local model packages, architecture-specific inference arguments, duplicate suppression, existing-output append behavior, and Image Analyst licensing. |
| Nearmap AI Pack: Building Footprints | https://help.nearmap.com/kb/articles/787-ai-pack-building-footprints | Roof-outline semantics explicitly documented for generations 1 through 5, regularized vector output, and fidelity score interpretation. Verify the actual export's generation. |
| Rasterio transforms | https://rasterio.readthedocs.io/en/stable/topics/transforms.html | Pixel-to-spatial coordinate transformation. |
| Shapely make_valid | https://shapely.readthedocs.io/en/stable/reference/shapely.make_valid.html | Geometry repair behavior, including possible nonpolygonal results. |

## Interpretation boundaries

- **Text, visual concepts, and instance prompts are distinct experiments.** A concept box provides an example of what to find. An instance box or point identifies an object to delineate. Evaluate the human input required by each.
- **Native Meta training uses a separate GPU environment.** The protocol targets Linux or WSL2 on Windows. No CUDA, driver, or training installation is implied by these documentation files. ArcGIS uses its own compatible environment.
- **Footprint semantics must be recorded.** A provider's layer name is insufficient evidence of ground-level geometry. Roof and ground references cannot be silently mixed.
- **Local results require local evidence.** Vendor benchmarks, model scores, synthetic tests, and archived imagery-download counts cannot substitute for held-out building extraction evaluation.
- **The data path is local.** No source above authorizes publication of Nearmap data or requires a paid API call. The workflow uses locally supplied imagery, labels, checkpoints, and optional vector exports.

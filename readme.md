This repository contains source code to reproduce the first part of the experiments described in the paper: 
"High Fidelity 3D Imaging of Dental Scenes using Gaussian Splatting"

Installation
- Install packages in `models/` using `pip install models/*`
- Install requirements in `requirements.txt` using `pip install -r requirements.txt`

Usage:
- Put video files in `DATA/raw/` directory
- Modify `scripts/1_extract_frames.py` to accommodate the video file name
- Run `python -m scripts.1_extract_frames` to extract frames from the video
- Run `python -m scripts.2_create_dataset` to create the dataset
- Run `python -m scripts.3_run_model <model_name>` to run the models
- Run `python -m scripts.4_results` to generate the results in csv format
- Run `python -m scripts.5_collect_render` to visualize the results

Note:
- To make the code self-contained, all models are included in the `models/` directory. We slightly modified the `nerfstudio` code to accommodate the train/test split.
- The code is tested on Python 3.10 with PyTorch 2.4 (CUDA 12.1)
- `nerfstudio` needs `gsplat-v1.0.0`, included is `gsplat-v1.4.0`. Thus, to run experiment on Instant-NGP/TensoRF, `gsplat-v1.0.0` should be installed, and the `gsplat-v1.4.0` should be installed when running on 3DGS/2DGS.

For more information, please refer to the paper.
```
@article{jin2025dentgs,
    author = "Jin, Chun-Xiao and Li, Meng-Xun and Yu, Huai and Gao, Yuan and Guo, Ya-Ping and Xia, Gui-Song and Huang, Cui",
    title = "High Fidelity 3D Imaging of Dental Scenes using Gaussian Splatting",
    journal = "Journal of Dental Research",
    year = "2025",
    publisher = "SAGE Publications Sage CA: Los Angeles, CA"
}
```
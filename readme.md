Contains source code to replicate the experiments for the paper: 
"High Fidelity 3D Imaging of Dental Scenes using Gaussian Splatting"

Installation
- Install packages in `models/` using `pip install models/*`
- Install requirements in `requirements.txt` using `pip install -r requirements.txt`

Usage:
- Put video files in `DATA/raw/` directory
- Modify `scripts/1_extract_frames.py` to change the video file name
- Run `python -m scripts.1_extract_frames` to extract frames from the video
- Run `python -m scripts.2_create_dataset` to create the dataset
- Run `python -m scripts.3_run_model <model_name>` to run the model
- Run `python -m scripts.4_results` to generate the results in csv format
- Run `python -m scripts.5_collect_render` to visualize the results

Note:
- nerfstudio needs gsplat-v1.0.0, included is gsplat-v1.4.0. Thus, to run experiment on Instant-NGP/TensoRF, gsplat-v1.0.0 should be installed, and the gsplat-v1.4.0 should be installed when running on 3DGS/2DGS.

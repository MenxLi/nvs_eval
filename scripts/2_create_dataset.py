from pathlib import Path
import subprocess, shutil

DATA_PATH = Path("DATA")
frame_dir_base = DATA_PATH / "frames"
dataset_dir = DATA_PATH / "dataset"

if __name__ == "__main__":
    dataset_dir.mkdir(parents=False, exist_ok=True)

    for frame_dir in frame_dir_base.iterdir():
        ds_name = frame_dir.name
        if frame_dir.is_dir():
            print(f"Processing {ds_name}")
            subprocess.check_call(
                [
                    "ns-process-data", "images", 
                    "--data", str(frame_dir), 
                    "--output-dir", str(dataset_dir / ds_name),
                    "--no-gpu"
                ]
            )

            shutil.copytree(dataset_dir/ds_name/'colmap', dataset_dir/ds_name/'llff')

            subprocess.check_call(
                [
                    "python3", "-m", "scripts.imgs2poses", 
                    str(dataset_dir / ds_name / 'llff'),
                ]
            )
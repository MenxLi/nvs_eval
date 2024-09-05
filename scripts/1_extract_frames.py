from pathlib import Path
import subprocess

DATA_PATH = Path("DATA")
video_path = DATA_PATH / "raw"
output_dir = DATA_PATH / "frames"

dataset_name_map = {
    "模型-后牙.mp4": "phantom_distal",
    "模型-前牙.mov": "phantom_anterior",
    "模型-头.MOV": "phantom_head",
    "真人-前牙.mov": "human_anterior",
    "真人-比色.MOV": "human_color",
    "真人-头.mp4": "human_head",
}

def convert_frames(video_path: Path, output_dir: Path):
    """
    Extract frames from a video file.
    """
    video_fname = video_path.name
    if video_fname not in dataset_name_map:
        raise ValueError(f"Unknown video file: {video_fname}")
    dataset_name = dataset_name_map[video_fname]
    output_dir = output_dir / dataset_name
    
    return subprocess.check_call(
        [
            "python", "-m", "scripts.video2image",
            str(video_path), str(output_dir), 
            "--frames", "110",
            "--auto-select"
        ]
    )

if __name__ == "__main__":
    output_dir.mkdir(parents=False, exist_ok=True)
    for video_path in video_path.iterdir():
        convert_frames(video_path, output_dir)
from pathlib import Path
from threading import Lock, Thread
import time, json, datetime
import argparse
from .utils.system_resource import GPU
from .utils import check_call

DATA_PATH = Path("DATA")
dataset_dir = DATA_PATH / "dataset"
output_dir = DATA_PATH / "output"

def get_exp_home(dataset_name: str, model_name: str) -> Path:
    return output_dir / dataset_name / model_name

def run_on_dataset(model_name: str, dataset_path: Path, device: int = 0):
    """
    Run the model on the dataset.
    - device: GPU device to use, CUDA_VISIBLE_DEVICES
    """
    cmds = [
        "ns-train", model_name, 
        "--data", str(dataset_path),
        "--output-dir", str(output_dir),
        "--vis", "viewer", 
        "--viewer.quit-on-train-completion", "True",
        "--timestamp", "exp"
    ]

    if model_name == 'instant-ngp':
        cmds.extend(["--pipeline.model.eval-num-rays-per-chunk", "4096"])
    if model_name == 'vanilla-nerf':
        cmds.extend([
            "--max-num-iterations", "1000000",
            "--mixed-precision", "False"
            ])
    if model_name == 'mipnerf':
        cmds.extend([
            "--max-num-iterations", "1000000",
            "--mixed-precision", "False"
            ])
    
    cmds.extend(["nerfstudio-data"])


    exp_home = get_exp_home(dataset_path.name, model_name)
    check_call(cmds, env={"CUDA_VISIBLE_DEVICES": str(device)}, error_log_file=exp_home / "error-train.txt")

def eval_on(model_name: str, dataset_name: str, device: int = 0):
    exp_home = get_exp_home(dataset_name, model_name)
    
    check_call(
        [
            "ns-eval", 
            "--load-config", str(exp_home / "exp" / "config.yml"),
            "--output-path", str(exp_home / "eval.json"),
            "--render-output-path", str(exp_home / "render"),
        ],
        env={"CUDA_VISIBLE_DEVICES": str(device)},
        error_log_file=exp_home / "error-eval.txt"
    )

if __name__ == "__main__":
    N_GPUS = 2

    parser = argparse.ArgumentParser(description="Run the model on the dataset")
    parser.add_argument("model_name", help="Name of the model to run")

    output_dir.mkdir(parents=False, exist_ok=True)
    args = parser.parse_args()
    model_name = args.model_name

    locks = [Lock() for _ in range(N_GPUS)]
    all_datasets = list(dataset_dir.iterdir())
    all_threads: list[Thread] = []

    def _run_thread(lock: Lock, dataset_path: Path, device: int):

        max_gpu_memory = 0
        running_flag = True

        def set_running_flag(flag: bool):
            nonlocal running_flag
            running_flag = flag

        def check_gpu_memory():
            nonlocal max_gpu_memory, running_flag

            gpu = GPU()
            while running_flag:
                gpu_device = gpu.query_device(device)
                if gpu_device is not None:
                    max_gpu_memory = max(max_gpu_memory, gpu_device.memory_used)
                time.sleep(1)

        def fn():
            ds_name = dataset_path.name
            exp_home = output_dir / ds_name / model_name

            def write_error_msg(msg: str):
                with open(exp_home / "error_{}.txt".format(datetime.datetime.now().strftime("%Y%m%d%H%M%S")), "w+") as f:
                    f.write(msg)

            with lock:

                print(f"Running {dataset_path.name} on GPU {device}")
                error_msg = None

                set_running_flag(True)
                t_gpu = Thread(target=check_gpu_memory)
                t_gpu.start()
                start_time = time.time()

                try:
                    run_on_dataset(model_name, dataset_path, device)
                except Exception as e:
                    write_error_msg(str(e.with_traceback(None)))
                    return
                finally:
                    end_time = time.time()
                    set_running_flag(False)
                    t_gpu.join()

                with open(exp_home / "log.json", "w") as f:
                    json.dump({
                        "gpu_memory": max_gpu_memory,
                        "time": end_time - start_time
                    }, f)
                
                try:
                    eval_on(model_name, ds_name, device)
                except Exception as e:
                    write_error_msg(str(e.with_traceback(None)))

        
        t = Thread(target=fn, daemon=True)
        t.start()
        all_threads.append(t)
        return t

    while all_datasets:
        for i in range(N_GPUS):
            if locks[i].locked():
                continue

            dataset_path = all_datasets.pop()
            _run_thread(locks[i], dataset_path, i)

        time.sleep(1)
    
    for t in all_threads:
        t.join()
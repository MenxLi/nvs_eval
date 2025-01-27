from pathlib import Path
from threading import Lock, Thread
import time, json, datetime, os
import argparse
from .utils.system_resource import GPU
from .utils import check_call

DATA_PATH = Path("DATA")
dataset_dir = DATA_PATH / "dataset"
output_dir = DATA_PATH / "output"

def get_exp_home(dataset_name: str, model_name: str) -> Path:
    return output_dir / dataset_name / model_name

def get_nerf_params(ds_name: str, factor: int = 8):
    cond_params = [
        "--expname", 'exp',
        "--basedir", str(output_dir/ds_name/"nerf"),
        "--datadir", str(dataset_dir/ds_name/"llff"),
        "--factor", str(factor),
        "--lrate", "5e-4",
    ]

    if (factor < 8):
        # otherwise OOM...
        cond_params += ["--no_batching"]

    # if ds_name == "phantom_anterior" or ds_name == "phantom_head" or ds_name == "human_head" or ds_name == 'human_color':
    #     cond_params += ["--lrate", "5e-3"]
    # else:
    #     cond_params += ["--lrate", "5e-4"]

    default_params = [
        "--dataset_type", "llff",
        "--llffhold", "10",
        "--N_rand", "1024",
        "--N_samples", "64",
        "--N_importance", "64",
        "--raw_noise_std", "1e0",
        "--i_video", "10000000",         # never do video rendering
        "--use_viewdirs",
        "--lrate_decay", "200",
    ]

    return cond_params + default_params

def run_on_dataset(model_name: str, dataset_path: Path, device: int = 0):
    """
    Run the model on the dataset.
    - device: GPU device to use, CUDA_VISIBLE_DEVICES
    """
    ds_name = dataset_path.name
    env = {"CUDA_VISIBLE_DEVICES": str(device)}

    if model_name == "nerf":
        # use orignial nerf impl.
        cmds = [
            "python", "models/nerf-pytorch/run_nerf_exp.py", 
        ] + get_nerf_params(ds_name, factor=2)
    
    elif model_name == "2dgs":
        cmds = [
            "python", "-m", "scripts.train_2dgs",
        ]
        env["EXP_NAME"] = ds_name
    elif model_name == "3dgs":
        cmds = [
            "python", "-m", "scripts.train_3dgs", "default"
        ]
        env["EXP_NAME"] = ds_name

    else:
        # use nerf studio
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
        cmds.extend(["nerfstudio-data"])

    exp_home = get_exp_home(dataset_path.name, model_name)
    check_call(cmds, env=env, error_log_file=exp_home / "error-train.txt")

def eval_on(model_name: str, dataset_name: str, device: int = 0):
    exp_home = get_exp_home(dataset_name, model_name)

    if model_name == "nerf":
        check_call(
            [ "python", "models/nerf-pytorch/run_nerf_exp.py", ]
            + get_nerf_params(dataset_name, factor=1) + 
            [
                "--render_only", 
                "--render_test", 
                "--render_factor", "1"
            ],
            env={"CUDA_VISIBLE_DEVICES": str(device)},
            error_log_file=exp_home / "error-eval.txt"
        )
    elif model_name == "2dgs" or model_name == "3dgs":
        pass
    else:
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
    CUDA_VISIBLE_DEVICES = os.getenv("CUDA_VISIBLE_DEVICES", "0")
    CUDA_VISIBLE_DEVICES = list(map(int, CUDA_VISIBLE_DEVICES.split(",")))
    N_GPUS = os.getenv("N_GPUS", len(CUDA_VISIBLE_DEVICES))

    usable_gpus = CUDA_VISIBLE_DEVICES[:N_GPUS]

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
                if (exp_home / "render").exists():
                    print(f"Skipping {dataset_path.name} on GPU {device}")
                    return

                print(f"Running {dataset_path.name} on GPU {device}")

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
                    raise e

        
        t = Thread(target=fn, daemon=True)
        t.start()
        all_threads.append(t)
        return t

    while all_datasets:
        for i in range(N_GPUS):
            if locks[i].locked():
                continue

            dev_idx = usable_gpus[i]
            dataset_path = all_datasets.pop()
            _run_thread(locks[i], dataset_path, dev_idx)

        time.sleep(1)
    
    for t in all_threads:
        t.join()
from pathlib import Path
import json
import pandas as pd
import dataclasses

DATA_PATH = Path("DATA")
output_dir = DATA_PATH / "output"

dataset_names = []
for ds_dir in output_dir.iterdir():
    if ds_dir.is_dir():
        dataset_names.append(ds_dir.name)

methods = [ 'nerf', 'tensorf', 'instant-ngp', 'splatfacto' ]

@dataclasses.dataclass
class Result:
    dataset_name: str
    method: str
    psnr: float
    ssim: float
    time: float
    train_memory: float
    render_fps: float

def get_data(ds_name: str, method):
    exp_dir = output_dir / ds_name / method

    eval_metrics = json.load(open(exp_dir / "eval.json"))['results']
    log = json.load(open(exp_dir / "log.json"))

    res = Result(
        dataset_name=ds_name,
        method=method,
        psnr=eval_metrics["psnr"],
        ssim=eval_metrics["ssim"],
        render_fps=eval_metrics["fps"], 
        time=log["time"]/60,                                # min
        train_memory=log["gpu_memory"]/1024/1024/1024,      # GB
    )

    return res

def resource_usage(data_list: list[Result]):
    raw = {}
    for method in methods:
        raw[method] = {
            "time": [],
            "memory": [],
            "fps": []
        }
        for data in data_list:
            if data.method == method:
                raw[method]["time"].append(data.time)
                raw[method]["memory"].append(data.train_memory)
                raw[method]["fps"].append(data.render_fps)
    # calculate average and std
    def avg(data: list):
        return sum(data)/len(data)
    def std(data: list):
        return (sum([(d - avg(data))**2 for d in data])/len(data))**0.5
    avg_res = {}
    for method in methods:
        avg_res[method] = {
            "time": (avg(raw[method]["time"]), std(raw[method]["time"])),
            "memory": (avg(raw[method]["memory"]), std(raw[method]["memory"])),
            "fps": (avg(raw[method]["fps"]), std(raw[method]["fps"]))
        }
    # to pd.DataFrame
    def get_str(res: tuple):
        return f"{res[0]:.2f}±{res[1]:.2f}"
    df = pd.DataFrame({
        "time": [get_str(avg_res[method]["time"]) for method in methods],
        "memory": [get_str(avg_res[method]["memory"]) for method in methods],
        "fps": [get_str(avg_res[method]["fps"]) for method in methods],
        })
    df.index = methods
    return df.T
                

if __name__ == "__main__":
    results = []
    for ds_name in dataset_names:
        for method in methods:
            try:
                res = get_data(ds_name, method)
                results.append(res)
            except Exception as e:
                print(f"Error processing {ds_name} {method}: {e}")

    df = pd.DataFrame([dataclasses.asdict(r) for r in results])
    df.to_csv(DATA_PATH/"results.csv", index=False, encoding='utf-8')
    print(df)

    df_res = resource_usage(results)
    df_res.to_csv(DATA_PATH/"resource_usage.csv", encoding='utf-8')
    print(df_res)
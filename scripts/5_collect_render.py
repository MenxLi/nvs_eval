from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

DATA_PATH = Path("DATA")
output_dir = DATA_PATH / "output"

dataset_names = []
dataset_names = []
for ds_dir in output_dir.iterdir():
    if ds_dir.is_dir():
        dataset_names.append(ds_dir.name)

methods = [ 'nerf', 'tensorf', 'instant-ngp', 'splatfacto' ]

def get_image_pair(ds_name: str, method: str, idx: int):
    exp_dir = output_dir / ds_name / method

    img_path = exp_dir / f"render/eval_img_{idx:04d}.png"
    img = plt.imread(img_path)
    im_w = img.shape[1]

    im1 = img[:, :im_w//2]
    im2 = img[:, im_w//2:]
    return im1, im2

def crop_resize(img: np.ndarray, size: tuple[int, int]):
    h, w = img.shape[:2]
    if h > w:
        pad = (h - w) // 2
        img = img[pad:pad+w, :]
    else:
        pad = (w - h) // 2
        img = img[:, pad:pad+h]
    return cv.resize(img, size)

def run_on_ds(ds_name: str):
    global render_compare_dir, render_sq_dir

    render_sq_ds_dir = render_sq_dir / ds_name
    render_sq_ds_dir.mkdir(parents=False, exist_ok=True)

    for i in range(10):
        print(f"Processing {ds_name} {i}")
        renders = []
        for method in methods:
            im_real, im_render = get_image_pair(ds_name, method, i)
            renders.append(im_render)
        r_shape = renders[0].shape

        im_real = cv.resize(im_real, (r_shape[1], r_shape[0]))
        renders = [cv.resize(r, (r_shape[1], r_shape[0])) for r in renders]

        render_im = np.concatenate(renders, axis=1)
        im = np.concatenate([im_real, render_im], axis=1)
        save_name = render_compare_dir / f"{ds_name}_{i}.png"
        plt.imsave(save_name, im)

        resized_im_real = crop_resize(im_real, (512, 512))
        resized_renders = [crop_resize(r, (512, 512)) for r in renders]

        def rotate_clockwise(im):
            return cv.rotate(im, cv.ROTATE_90_CLOCKWISE)
        def rotate_anticlockwise(im):
            return cv.rotate(im, cv.ROTATE_90_COUNTERCLOCKWISE)
        
        if ds_name in ["human_anterior", "human_color", ]:
            resized_im_real = rotate_anticlockwise(resized_im_real)
            resized_renders = [rotate_anticlockwise(r) for r in resized_renders]

        plt.imsave(render_sq_ds_dir / f"real_{i}.png", resized_im_real)
        for j, r in enumerate(resized_renders):
            plt.imsave(render_sq_ds_dir / f"{methods[j]}_{i}.png", r)

    
    
if __name__ == "__main__":
    render_dir = DATA_PATH / "result_renders"
    render_dir.mkdir(parents=False, exist_ok=True)

    render_compare_dir = render_dir / "compare"
    render_sq_dir = render_dir / "square"

    render_compare_dir.mkdir(parents=False, exist_ok=True)
    render_sq_dir.mkdir(parents=False, exist_ok=True)
    
    ThreadPoolExecutor().map(run_on_ds, dataset_names)
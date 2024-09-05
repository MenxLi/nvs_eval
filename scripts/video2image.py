import argparse
import os, multiprocessing, time
import cv2 as cv
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


def clear_score(im):
    """
    Returns a score indicating how blurry the image is.
    The more blurry the image, the lower the score.
    """
    return cv.Laplacian(im, cv.CV_64F).var()

def extract_frames(video_path, output_dir, frames=100, auto_select=False):
    print("Extracting frames from video: ", video_path)
    print("Aim to extract: ", frames, "frames")

    video = cv.VideoCapture(video_path)

    if not video.isOpened():
        print("Error opening video file")
        return

    os.makedirs(output_dir, exist_ok=True)

    frame_count = 0
    extract_frame_count = 0

    if frames > 0:
        interval = int(video.get(cv.CAP_PROP_FRAME_COUNT) / frames)
    else:
        interval = 1
    print("Frame interval: ", interval)
    
    def process_frame(frame):
        score = clear_score(frame)
        return frame, score

    # Put the function of calculating the image clarity and writing to disk into a separate thread
    with ThreadPoolExecutor(
        max_workers = 8 if multiprocessing.cpu_count() > 8 else multiprocessing.cpu_count() - 1
        ) as executor:

        futures = []

        bar = tqdm(total=video.get(cv.CAP_PROP_FRAME_COUNT), desc="Extracting frames")
        while True:
            ret, frame = video.read()
            if not ret:
                break

            if not auto_select:
                if frame_count % interval == 0:
                    output_path = os.path.join(output_dir, f"{extract_frame_count}.jpg")
                    cv.imwrite(output_path, frame)
                    extract_frame_count += 1
            else:
                futures.append(executor.submit(process_frame, frame))
                if frame_count % interval == 0:
                    batch = [future.result() for future in as_completed(futures)]

                    max_score_index = max(range(len(batch)), key=lambda i: batch[i][1])
                    output_path = os.path.join(output_dir, f"{extract_frame_count}.jpg")

                    executor.submit(cv.imwrite, output_path, batch[max_score_index][0])
                    extract_frame_count += 1

                    futures.clear()

            frame_count += 1
            bar.update(1)

    video.release()

    print(f"Successfully extracted {extract_frame_count} frames out of {frame_count}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from a video file")
    parser.add_argument("video_path", help="Path to the video file")
    parser.add_argument("output_dir", help="Path to the output directory")
    parser.add_argument("--frames", type=int, default=100, help="Number of frames to extract")
    parser.add_argument("--auto-select", action='store_true', help="Automatically select the best frames to extract")
    args = parser.parse_args()

    _s = time.time()
    extract_frames(args.video_path, args.output_dir, args.frames, args.auto_select)
    print("Time taken for frame extraction: ", time.time() - _s)
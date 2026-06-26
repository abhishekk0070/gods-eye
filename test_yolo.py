import time
import cv2
from ultralytics import YOLO
import torch

# # Optional: use all CPU cores
# torch.set_num_threads(4)

# Path to any test image
IMAGE_PATH = "ffff.jpg"

# Models to compare
MODELS = [
    "yolov8n.pt",
    "yolo11n.pt",
    "yolo26n.pt"
]

# Load image
frame = cv2.imread(IMAGE_PATH)

if frame is None:
    raise FileNotFoundError(f"Could not load image: {IMAGE_PATH}")

# Resize to the same size used in your application
frame = cv2.resize(frame, (960, 540))

for model_name in MODELS:
    print(f"\nBenchmarking {model_name}")

    # Load model
    model = YOLO(model_name)
    model.to("cpu")

    # Warmup (ignore first few runs)
    print("Warming up...")
    for _ in range(10):
        model(frame, classes=[0], verbose=False)

    # Actual benchmark
    num_runs = 100

    start = time.perf_counter()

    for _ in range(num_runs):
        model(frame, classes=[0], verbose=False)

    end = time.perf_counter()

    total_time = end - start
    avg_time = total_time / num_runs
    fps = 1 / avg_time

    print(f"Average inference time: {avg_time:.4f} seconds")
    print(f"Average FPS: {fps:.2f}")
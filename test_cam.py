import os

os.environ["KMP_WARNINGS"] = "0"
import warnings

warnings.filterwarnings("ignore")

import cv2
from ultralytics import YOLO

# Load the small pretrained YOLOv8 model (downloads ~6MB on first run)
model = YOLO("yolov8n.pt")

# RTSP_URL = "rtsp://admin:index123@10.157.88.226:554"  # 👈 put your tested URL here
RTSP_URL = "rtsp://admin:1234@192.168.2.199"  # 👈 put your tested URL here

cap = cv2.VideoCapture(RTSP_URL)

if not cap.isOpened():
    print("Failed to open stream")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame read failed")
        break

    frame = cv2.resize(frame, (960, 540))  # 👈 added

    results = model(frame, classes=[0], verbose=False)
    annotated = results[0].plot()

    cv2.imshow("Intrusion Detection - Test", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

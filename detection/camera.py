import os
os.environ["PYTORCH_DISABLE_NNPACK"] = "1"

import cv2
from ultralytics import YOLO
import threading
import time
from collections import deque
from datetime import datetime
from django.conf import settings

RTSP_URL = os.environ.get("RTSP_URL")
YOLO_MODEL_PATH = os.environ.get("YOLO_MODEL_PATH", "yolo26n.pt")

FPS_ESTIMATE = 25
YOLO_INTERVAL = 0.33     # minimum gap between YOLO runs (~3 FPS) - inference runs in its own thread

CLIP_DURATION_SECONDS = 10
PRE_ROLL_SECONDS = 0     # start clips when the person is detected
POST_ROLL_SECONDS = CLIP_DURATION_SECONDS - PRE_ROLL_SECONDS
GAP_TOLERANCE_SECONDS = 1.0  # how long a person can be briefly "lost" before we treat the event as over
PRE_BUFFER_SECONDS = 1.5

ZONE_REFRESH_SECONDS = 5  # how often to re-check the DB for an updated detection zone

# Helps reduce RTSP buffering/latency drift over time
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")


def point_in_polygon(x, y, polygon):
    """
    Standard ray-casting point-in-polygon test.
    polygon: list of (x, y) tuples, same coordinate space as (x, y).
    Returns True if (x, y) is inside the polygon.
    """
    n = len(polygon)
    if n < 3:
        return True  # no real polygon defined -- treat as "no restriction"

    inside = False
    x1, y1 = polygon[0]
    for i in range(1, n + 1):
        x2, y2 = polygon[i % n]
        if (y < y1) != (y < y2):
            if y2 != y1:
                x_intersect = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
                if x < x_intersect:
                    inside = not inside
        x1, y1 = x2, y2
    return inside


class CameraStream:
    def __init__(self):
        if not RTSP_URL:
            raise RuntimeError("RTSP_URL environment variable is required")

        self.model = YOLO(YOLO_MODEL_PATH)
        self.model.to("cpu")

        self.cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        # Raw frame from capture thread (pre-boxes), protected by frame_lock
        self.latest_raw_frame = None
        self.frame_lock = threading.Lock()

        # Clean frame for the LIVE FEED only — no boxes drawn on this one
        self.latest_live_frame = None
        self.live_lock = threading.Lock()

        # Rolling pre-buffer: (timestamp, frame) tuples, last PRE_BUFFER_SECONDS
        self.pre_buffer = deque()
        self.pre_buffer_lock = threading.Lock()

        self.running = True

        # Recording / event state
        self.is_recording = False
        self.writer = None
        self.writer_lock = threading.Lock()
        self.last_seen_time = None          # last timestamp a person was actually seen by YOLO
        self.person_present = False         # are we currently inside an active "event" (incl. gap tolerance)?
        self.recording_stop_time = None       # fixed stop time for 10-second clips
        self.next_recording_frame_time = None
        self.current_clip_path = None
        self.max_confidence = 0
        self.recording_start_time = None
        self.thumbnail_frame = None
        self.thumbnail_boxes = []

        # Last known YOLO boxes, used only for thumbnails.
        self._last_boxes = []
        self.last_detection_time = 0

        # Detection zone (polygon), in PIXEL coordinates matching the resized frame (960x540).
        # Empty list = no restriction, detect anywhere in frame (original behavior).
        self._zone_polygon = []
        self._zone_lock = threading.Lock()
        self._last_zone_check = 0
        self._load_zone()  # initial load before threads start

        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.inference_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self.capture_thread.start()
        self.inference_thread.start()

    # ---------------------------------------------------------------------
    # Detection zone
    # ---------------------------------------------------------------------

    def _load_zone(self):
        """
        Reload the active detection zone from the DB. Points are stored normalized
        (0.0-1.0) and converted here to pixel coordinates for the 960x540 working
        frame size used throughout this class.
        """
        try:
            from detection.models import DetectionZone
            zone = DetectionZone.objects.order_by('-updated_at').first()
            if zone and zone.points and len(zone.points) >= 3:
                polygon = [(float(px) * 960.0, float(py) * 540.0) for (px, py) in zone.points]
            else:
                polygon = []
        except Exception as e:
            print(f"[ZONE ERROR] Failed to load detection zone: {e}")
            polygon = []

        with self._zone_lock:
            self._zone_polygon = polygon

    def _maybe_refresh_zone(self):
        """Called periodically from the inference loop so zone edits apply without a restart."""
        now = time.time()
        if now - self._last_zone_check >= ZONE_REFRESH_SECONDS:
            self._last_zone_check = now
            self._load_zone()

    def _get_zone_polygon(self):
        with self._zone_lock:
            return list(self._zone_polygon)

    def _box_in_zone(self, x1, y1, x2, y2, polygon):
        """A detection counts if its box CENTER falls inside the zone (or there is no zone)."""
        if not polygon:
            return True
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        return point_in_polygon(cx, cy, polygon)

    def _draw_boxes(self, frame, boxes):
        """Draw detection boxes onto a fresh frame for thumbnails only."""
        out = frame.copy()
        for (x1, y1, x2, y2, conf, label) in boxes:
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(out, f"{label} {conf:.2f}", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return out

    # ---------------------------------------------------------------------
    # Recording lifecycle
    # ---------------------------------------------------------------------

    def _start_recording(self, first_frame):
        """Open the writer when a person is detected."""
        clips_dir = os.path.join(settings.MEDIA_ROOT, 'clips')
        os.makedirs(clips_dir, exist_ok=True)

        filename = datetime.now().strftime("%Y-%m-%d_%H%M%S") + ".mp4"
        filepath = os.path.join(clips_dir, filename)

        h, w = first_frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(filepath, fourcc, FPS_ESTIMATE, (w, h))

        self.is_recording = True
        self.recording_start_time = time.time()
        self.recording_stop_time = self.last_seen_time + POST_ROLL_SECONDS
        self.next_recording_frame_time = self.last_seen_time
        self.current_clip_path = f"clips/{filename}"
        self.max_confidence = 0
        print(f"[REC START] {filepath}")

        if PRE_ROLL_SECONDS <= 0:
            return

        # Write the pre-roll as clean frames. We resample by nearest timestamp so the
        # beginning of the clip does not reuse one stale frame for long stretches.
        anchor = self.last_seen_time
        start_t = anchor - PRE_ROLL_SECONDS

        with self.pre_buffer_lock:
            buffered = list(self.pre_buffer)  # [(ts, frame), ...] oldest -> newest

        relevant = [b for b in buffered if start_t <= b[0] <= anchor]

        if relevant:
            frame_interval = 1.0 / FPS_ESTIMATE
            expected_shape = first_frame.shape
            slot_count = max(1, int(round(PRE_ROLL_SECONDS * FPS_ESTIMATE)))

            buf_idx = 0
            for slot in range(slot_count):
                target_t = start_t + slot * frame_interval
                while (buf_idx + 1) < len(relevant) and relevant[buf_idx + 1][0] < target_t:
                    buf_idx += 1
                candidates = [relevant[buf_idx]]
                if (buf_idx + 1) < len(relevant):
                    candidates.append(relevant[buf_idx + 1])
                ts, f = min(candidates, key=lambda item: abs(item[0] - target_t))

                if f is None or f.shape != expected_shape:
                    print(f"[REC WARN] Skipping malformed pre-roll frame (shape={getattr(f, 'shape', None)})")
                    continue
                try:
                    with self.writer_lock:
                        if self.writer is not None:
                            self.writer.write(f)
                except Exception as e:
                    print(f"[REC WARN] Failed writing pre-roll frame: {e}")

    def _wait_for_file_stable(self, path, timeout=30, interval=0.5):
        deadline = time.time() + timeout
        last_size = -1
        while time.time() < deadline:
            if os.path.exists(path):
                current_size = os.path.getsize(path)
                if current_size > 0 and current_size == last_size:
                    return True
                last_size = current_size
            time.sleep(interval)
        return False

    def _stop_recording(self):
        """Release writer and spin up a background thread for ffmpeg + thumbnail + DB."""
        with self.writer_lock:
            if self.writer:
                self.writer.release()
                self.writer = None

        clip_path = self.current_clip_path
        duration = CLIP_DURATION_SECONDS
        confidence = self.max_confidence
        thumbnail_frame = self.thumbnail_frame.copy() if self.thumbnail_frame is not None else None
        thumbnail_boxes = list(self.thumbnail_boxes)

        self.is_recording = False
        self.current_clip_path = None
        self.recording_stop_time = None
        self.next_recording_frame_time = None
        self._last_boxes = []
        self.thumbnail_frame = None
        self.thumbnail_boxes = []

        t = threading.Thread(
            target=self._finalize_clip,
            args=(clip_path, duration, confidence, thumbnail_frame, thumbnail_boxes),
            daemon=True
        )
        t.start()

    def _finalize_clip(self, clip_path, duration, confidence, thumbnail_frame, thumbnail_boxes):
        """Runs in background thread - convert, thumbnail, save to DB."""
        import subprocess

        raw_path = os.path.join(settings.MEDIA_ROOT, clip_path)
        h264_path = raw_path.replace('.mp4', '_h264.mp4')

        try:
            result = subprocess.run([
                'ffmpeg', '-y',
                '-i', raw_path,
                '-vcodec', 'libx264',
                '-crf', '28',
                '-preset', 'ultrafast',
                h264_path
            ], capture_output=True)
            if result.returncode != 0:
                print(f"[FFMPEG ERROR] {result.stderr.decode()}")
                return
            os.remove(raw_path)
            os.rename(h264_path, raw_path)
            print(f"[FFMPEG] Converted: {raw_path}")
        except Exception as e:
            print(f"[FFMPEG ERROR] {e}")
            return

        thumbnail_path = ''
        try:
            thumbnails_dir = os.path.join(settings.MEDIA_ROOT, 'thumbnails')
            os.makedirs(thumbnails_dir, exist_ok=True)

            stable = self._wait_for_file_stable(raw_path, timeout=30, interval=0.5)
            if not stable:
                print(f"[THUMB ERROR] File never stabilised: {raw_path}")
            else:
                if thumbnail_frame is None:
                    cap = cv2.VideoCapture(raw_path)
                    cap.set(cv2.CAP_PROP_POS_MSEC, PRE_ROLL_SECONDS * 1000)
                    ret, thumb_frame = cap.read()
                    cap.release()
                else:
                    ret = True
                    thumb_frame = self._draw_boxes(thumbnail_frame, thumbnail_boxes)

                if ret and thumb_frame is not None:
                    thumb_filename = os.path.basename(raw_path).replace('.mp4', '.jpg')
                    thumb_full_path = os.path.join(thumbnails_dir, thumb_filename)
                    cv2.imwrite(thumb_full_path, thumb_frame)
                    thumbnail_path = f'thumbnails/{thumb_filename}'
                    print(f"[THUMB] Saved {thumb_full_path}")
                else:
                    print(f"[THUMB ERROR] Could not read frame from: {raw_path}")
        except Exception as e:
            print(f"[THUMB ERROR] {e}")

        try:
            from detection.models import Event
            Event.objects.create(
                clip_path=clip_path,
                thumbnail_path=thumbnail_path,
                confidence=confidence,
                duration_seconds=round(duration, 1)
            )
            print(f"[REC STOP] Saved event, duration={duration:.1f}s")
        except Exception as e:
            print(f"[DB ERROR] {e}")

    # ---------------------------------------------------------------------
    # Main loops
    # ---------------------------------------------------------------------

    def _capture_loop(self):
        """
        Runs continuously, never blocked by YOLO inference.
        - Always updates the clean live-feed frame (no boxes).
        - Always maintains the rolling pre-buffer of raw frames.
        - If a recording is active, writes clean frames to the clip.
        """
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            frame = cv2.resize(frame, (960, 540))
            now = time.time()

            with self.frame_lock:
                self.latest_raw_frame = frame

            # Live feed: always clean, no boxes
            with self.live_lock:
                self.latest_live_frame = frame

            # Rolling pre-buffer, trimmed to PRE_BUFFER_SECONDS.
            with self.pre_buffer_lock:
                self.pre_buffer.append((now, frame.copy()))
                while self.pre_buffer and (now - self.pre_buffer[0][0]) > PRE_BUFFER_SECONDS:
                    self.pre_buffer.popleft()

            # Recording: write clean frame, sampled to match FPS_ESTIMATE.
            if self.is_recording and self.writer is not None:
                frame_interval = 1.0 / FPS_ESTIMATE
                if self.next_recording_frame_time is None:
                    self.next_recording_frame_time = now

                write_until = now
                if self.recording_stop_time is not None:
                    write_until = min(write_until, self.recording_stop_time)

                while (
                    self.next_recording_frame_time < write_until
                    and now >= self.next_recording_frame_time
                ):
                    try:
                        with self.writer_lock:
                            if self.writer is not None:
                                self.writer.write(frame)
                    except Exception as e:
                        print(f"[REC WARN] Failed writing live frame to clip: {e}")
                    self.next_recording_frame_time += frame_interval

                if self.recording_stop_time is not None and now >= self.recording_stop_time:
                    self._stop_recording()

    def _inference_loop(self):
        """
        Runs YOLO at its own pace, reading the latest raw frame.
        Starts a fixed 10-second clip at the first detection INSIDE the active zone
        (or anywhere in frame if no zone is configured).
        """
        while self.running:
            self._maybe_refresh_zone()

            with self.frame_lock:
                frame = self.latest_raw_frame.copy() if self.latest_raw_frame is not None else None

            if frame is None:
                time.sleep(0.05)
                continue

            loop_start = time.time()

            results = self.model(frame, classes=[0, 2], verbose=False)  # 0=person, 2=car
            polygon = self._get_zone_polygon()

            new_boxes = []
            in_zone_confidences = []
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                label = results[0].names[cls_id]

                if not self._box_in_zone(x1, y1, x2, y2, polygon):
                    continue  # detected, but outside the active zone -- ignore

                new_boxes.append((x1, y1, x2, y2, conf, label))
                in_zone_confidences.append(conf)

            self._last_boxes = new_boxes
            person_detected = len(new_boxes) > 0

            now = time.time()

            if person_detected:
                self.last_seen_time = now
                self.person_present = True

                if not self.is_recording:
                    self._start_recording(frame)

                if in_zone_confidences:
                    best_conf = max(in_zone_confidences)
                    if best_conf >= self.max_confidence:
                        self.max_confidence = best_conf
                        self.thumbnail_frame = frame.copy()
                        self.thumbnail_boxes = new_boxes

            elif self.person_present and self.last_seen_time is not None:
                gap = now - self.last_seen_time
                if gap >= GAP_TOLERANCE_SECONDS:
                    self.person_present = False

            elapsed = time.time() - loop_start
            sleep_time = YOLO_INTERVAL - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def get_jpeg_frame(self):
        """Live feed — always clean, no detection boxes."""
        with self.live_lock:
            if self.latest_live_frame is None:
                return None
            ret, jpeg = cv2.imencode('.jpg', self.latest_live_frame)
            return jpeg.tobytes() if ret else None

    def stop(self):
        """Optional clean shutdown."""
        self.running = False
        if self.writer:
            self._stop_recording()
        if self.cap:
            self.cap.release()


_camera_stream_instance = None

def get_camera_stream():
    global _camera_stream_instance
    if _camera_stream_instance is None:
        _camera_stream_instance = CameraStream()
    return _camera_stream_instance

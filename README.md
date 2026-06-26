# Gods Eye

Django-based CCTV person detection dashboard using OpenCV and Ultralytics YOLO.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Set your local values in `.env`, then export them before running:

```bash
export DJANGO_SECRET_KEY="replace-this-with-a-secret-key"
export DJANGO_DEBUG=1
export DJANGO_ALLOWED_HOSTS="localhost,127.0.0.1"
export RTSP_URL="rtsp://username:password@camera-ip-address"
export YOLO_MODEL_PATH="yolo26n.pt"
```

Place the YOLO weights file at the path in `YOLO_MODEL_PATH`, then run:

```bash
python manage.py migrate
python manage.py runserver
```

Recorded clips and thumbnails are stored under `media/`, which is intentionally ignored by git.

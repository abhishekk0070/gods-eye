from django.http import StreamingHttpResponse
from django.shortcuts import render
from .camera import get_camera_stream
from .cleanup import start_cleanup_thread
from .models import Event, DetectionZone
import time
import json
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_protect


def index(request):
    start_cleanup_thread()
    events = Event.objects.all()[:10]  # last 10 events
    context = {
        'events': events
    }
    return render(request, 'detection/index.html', context)

def gen_frames():
    camera_stream = get_camera_stream()
    while True:
        frame = camera_stream.get_jpeg_frame()
        if frame is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.05)

def video_feed(request):
    return StreamingHttpResponse(
        gen_frames(),
        content_type='multipart/x-mixed-replace; boundary=frame'
    )


def events_json(request):
    events = Event.objects.all()[:10]
    data = []
    for e in events:
        data.append({
            'id': e.id,
            'timestamp': e.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'clip_path': e.clip_path,
            'thumbnail_path': e.thumbnail_path,
            'confidence': round(e.confidence, 2),
            'duration_seconds': e.duration_seconds,
        })
    return JsonResponse({'events': data})


def zone_json(request):
    """Returns the currently active detection zone, if any."""
    zone = DetectionZone.objects.order_by('-updated_at').first()
    if zone:
        return JsonResponse({'points': zone.points, 'name': zone.name})
    return JsonResponse({'points': [], 'name': None})


@csrf_protect
@require_http_methods(['POST'])
def zone_save(request):
    """
    Save (replace) the active detection zone.
    Expects JSON body: {"points": [[x, y], [x, y], ...]} with x/y normalized 0.0-1.0.
    Sending an empty points list clears the zone (detect anywhere in frame).
    """
    try:
        body = json.loads(request.body.decode('utf-8'))
        points = body.get('points', [])

        if points:
            if not isinstance(points, list) or len(points) < 3:
                return JsonResponse({'error': 'A zone needs at least 3 points'}, status=400)
            for p in points:
                if not (isinstance(p, list) and len(p) == 2):
                    return JsonResponse({'error': 'Each point must be [x, y]'}, status=400)
                x, y = p
                if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                    return JsonResponse({'error': 'Points must be normalized between 0.0 and 1.0'}, status=400)

        # Single active zone: clear existing rows, save the new one
        DetectionZone.objects.all().delete()
        zone = DetectionZone.objects.create(name='default', points=points)

        return JsonResponse({'ok': True, 'points': zone.points})
    except (json.JSONDecodeError, ValueError) as e:
        return JsonResponse({'error': f'Invalid request: {e}'}, status=400)



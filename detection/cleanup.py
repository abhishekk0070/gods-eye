import os
import threading
import time
from datetime import timedelta
from django.conf import settings
from django.utils import timezone
from detection.models import Event

CLEANUP_CHECK_INTERVAL_SECONDS = 30 * 60    # how often to check for old events (every 5 min)
RETENTION_HOURS = 24                        # delete events older than this


def _cleanup_once():
    """
    Find Event rows older than RETENTION_HOURS, delete their clip + thumbnail
    files from disk (if present), then delete the DB rows. Each step is best-effort:
    a missing file or a failed delete on one event should not stop the rest from
    being processed.
    """
    
    cutoff = timezone.now() - timedelta(hours=RETENTION_HOURS)
    old_events = Event.objects.filter(timestamp__lt=cutoff)

    count = old_events.count()
    if count == 0:
        return

    deleted = 0
    for event in old_events:
        for rel_path in (event.clip_path, event.thumbnail_path):
            if not rel_path:
                continue
            full_path = os.path.join(settings.MEDIA_ROOT, rel_path)
            try:
                if os.path.exists(full_path):
                    os.remove(full_path)
            except Exception as e:
                print(f"[CLEANUP WARN] Could not delete file {full_path}: {e}")

        try:
            event.delete()
            deleted += 1
        except Exception as e:
            print(f"[CLEANUP WARN] Could not delete Event id={event.id}: {e}")

    print(f"[CLEANUP] Removed {deleted}/{count} events older than {RETENTION_HOURS}h")


def _cleanup_loop():
    while True:
        try:
            _cleanup_once()
        except Exception as e:
            # Catch-all so a single bad run (e.g. transient DB issue) never kills the thread
            print(f"[CLEANUP ERROR] {e}")
        time.sleep(CLEANUP_CHECK_INTERVAL_SECONDS)


_cleanup_thread_started = False
_cleanup_thread_lock = threading.Lock()


def start_cleanup_thread():
    """
    Starts the background cleanup loop exactly once per process.
    Safe to call multiple times (e.g. once per request) -- only the first call
    actually starts the thread.
    """
    global _cleanup_thread_started
    with _cleanup_thread_lock:
        if _cleanup_thread_started:
            return
        _cleanup_thread_started = True
        t = threading.Thread(target=_cleanup_loop, daemon=True)
        t.start()
        print(f"[CLEANUP] Background cleanup thread started "
              f"(checking every {CLEANUP_CHECK_INTERVAL_SECONDS // 60} min, "
              f"retention {RETENTION_HOURS}h)")

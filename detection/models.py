from django.db import models


class Event(models.Model):
    timestamp = models.DateTimeField(auto_now_add=True)
    clip_path = models.CharField(max_length=255)
    thumbnail_path = models.CharField(max_length=255, blank=True, default='')
    confidence = models.FloatField()
    duration_seconds = models.FloatField(default=0)

    class Meta:
        ordering = ['-timestamp']

    def __str__(self):
        return f"Person detected at {self.timestamp}"


class DetectionZone(models.Model):
    """
    Stores a single active detection zone as a polygon of normalized points
    (each x/y in the range 0.0-1.0, relative to frame width/height) so the
    zone stays correct regardless of the camera's actual resolution.

    Only one row is expected to exist at a time -- saving a new zone should
    replace the previous one. An empty/missing zone means "detect anywhere
    in frame" (the original behavior).
    """
    name = models.CharField(max_length=100, default='default')
    # List of [x, y] pairs, each 0.0-1.0, e.g. [[0.1, 0.2], [0.8, 0.2], [0.8, 0.9], [0.1, 0.9]]
    points = models.JSONField(default=list)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Zone '{self.name}' ({len(self.points)} points)"
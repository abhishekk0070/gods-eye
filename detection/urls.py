from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('video_feed/', views.video_feed, name='video_feed'),
    path('events.json', views.events_json, name='events_json'),
    path('zone.json', views.zone_json, name='zone_json'),
    path('zone/save', views.zone_save, name='zone_save'),
]
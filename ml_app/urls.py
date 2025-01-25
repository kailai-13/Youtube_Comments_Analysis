from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('youtube/sentiment/', views.sentiment_analysis, name='sentiment_analysis'),
    path('youtube/hate_speech/', views.hate_speech_analysis, name='hate_speech_analysis'),
]

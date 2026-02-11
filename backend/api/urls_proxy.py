"""URL patterns for the OpenAI-compatible proxy API."""
from django.urls import path

from . import views_proxy

urlpatterns = [
    path('chat/completions', views_proxy.chat_completions, name='v1-proxy-chat'),
]

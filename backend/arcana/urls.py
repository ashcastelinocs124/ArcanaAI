"""Root URL configuration for arcana project."""
from django.urls import include, path

urlpatterns = [
    path('', include('api.urls')),
]

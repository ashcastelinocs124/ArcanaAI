"""Root URL configuration for arcana project."""
from django.urls import include, path

urlpatterns = [
    path('api/v1/', include('api.urls_v1')),
    path('', include('api.urls')),
]

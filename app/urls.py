from django.urls import path
from .views import *

urlpatterns = [
    path('', predict_image, name='predict_image'),
    ]

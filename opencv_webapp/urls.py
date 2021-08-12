from django.urls import include, path
from . import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
 path('', views.first_view, name='first_view'),
 path('uimage/', views.uimage, name='uimage'),
 path('dface/', views.dface, name='dface'),
 path('home/', views.home, name='home'),
 path('uimage/auto', views.ajustar, name='auto'),
 path('uimage/auto1', views.puntos, name='auto1'),
 path('uimage/auto2', views.Adaptative, name='auto2'),
 path('uimage/auto3', views.gris, name='auto3'),
 path('uimage/auto4', views.giro, name='auto4'),
 path('uimage/auto5', views.color, name='auto5'),
]

urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

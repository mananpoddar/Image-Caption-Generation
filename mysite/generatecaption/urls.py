from django.conf.urls import url
from generatecaption import views
app_name = "generatecaption"

urlpatterns = [
    url(r'^$', views.index, name='index'),
]
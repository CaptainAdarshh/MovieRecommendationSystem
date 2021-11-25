from django.conf.urls import url
from appone import views

urlpatterns = [
    url('^$', views.HomePage.as_view(), name='index'),
    url(r'^movieplot/',views.MoviePlot.as_view(), name='movieplot'),
    url(r'^moviekey/', views.MovieKey.as_view(), name='moviekey'),
    url(r'^toptenvoted/', views.TopTenVoted.as_view(), name='toptenvoted'),
    url(r'^toptenpopular/', views.TopTenPopular.as_view(), name='toptenpopular'),
]

"""
URL configuration for myWeb project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from webs import views

urlpatterns = [
    path('', views.login),
    path('index/', views.index, name='index'),
    path('register/', views.register, name='register'),
    path('tutorial/', views.tutorial, name='tutorial'),
    path('test/', views.test, name='test'),

    path('tutorialClass/', views.tutorialClass, name='tutorialClass'),
    path('tutorialGreet/', views.tutorialGreet, name='tutorialGreet'),
    path('tutorialMeat/', views.tutorialMeat, name='tutorialMeat'),
    path('tutorialLeisure/', views.tutorialLeisure, name='tutorialLeisure'),
    path('tutorialRelation/', views.tutorialRelation, name='tutorialRelation'),

    path('testClass', views.testClass, name='testClass'),
    path('testGreet', views.testGreet, name='testGreet'),
    path('testMeat', views.testMeat, name='testMeat'),
    path('testLeisure', views.testLeisure, name='testLeisure'),
    path('testRelation', views.testRelation, name='testRelation'),
    path('admin/', admin.site.urls),
]

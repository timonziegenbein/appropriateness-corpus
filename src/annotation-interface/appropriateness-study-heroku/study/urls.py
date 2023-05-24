"""study URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.0/topics/http/urls/
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
from django.urls import path, include
from . import views

app_name = "study"
urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.index, name='index'),
    path('login', views.login, name='login'),
    path('logout', views.logout, name='logout'),
    path('dashboard', views.dashboard, name='dashboard'), 
    path('annotate/<str:post_id>/', views.annotate, name='annotate'), 
    path('save_annotation', views.save_annotation, name='save_annotation'),
    path('upload_posts', views.upload_posts, name='upload_posts'),
    path('create_show_users', views.create_show_users, name='create_show_users'),
    path('download_annotations', views.download_annotations, name='download_annotations'),
    path('download_users', views.download_users, name='download_users'),
    path('download_posts', views.download_posts, name='download_posts'),
    path('view_annotations/<int:batch_num>/', views.view_annotations, name='view_annotations'),
    path('deactivate_user/<int:user_id>/', views.deactivate_user, name='deactivate_user'),
    path('activate_user/<int:user_id>/', views.activate_user, name='activate_user'),
    path('download_annotations_view/<int:batch_num>/', views.download_annotations_view, name='download_annotations_view'),
]

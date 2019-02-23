from django.urls import path, re_path
from django.conf.urls import url
from . import views

urlpatterns =[
    path("hello/", views.hello, name='hello'),
    path("", views.frontpage),
    #re_path(r'^settings$', views.settings)
    path('settings/', views.settings),
    #re_path(r'^add_category$', views.addCategory),
    path('add_category', views.addCategory), #it works
    #re_path(r'^delete_category/{{category.category}}', views.deleteCategory), #type1 no work
    #url(r'delete_category/{{ category.category }}', views.deleteCategory), #type1 no work
    re_path(r'^delete_category/(?P<category>\w+)', views.deleteCategory), #type 2 it works
    #url(r'delete_category/(?P<category>\w+)', views.deleteCategory), #tutorial type it works
    path('add_record', views.addRecord), #works
    #re_path(r'add_record$', views.addRecord) #tutorial type
    path('delete_record', views.deleteRecord)

]
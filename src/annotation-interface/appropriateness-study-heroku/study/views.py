import ast
from django.contrib import messages
from django.contrib.auth.decorators import user_passes_test, permission_required
from django.shortcuts import render, redirect

from django.contrib.auth import authenticate
from django.utils import timezone
from datetime import datetime
from . import PAGES, controller, models


import pandas as pd
import csv, io


### Helper Function
def set_context(request):
    context = {
        'username': request.session['username'],
        'fullname': request.session['fullname'],
        'is_staff': request.session['is_staff'],
        'batch': request.session['batch'],
    }
    return context

def is_logged_in(request):
    return 'username' in request.session.keys() and request.session['username'] is not None

def is_batch_assigned(request):
    return 'batch' in request.session.keys() and request.session['batch'] is not None

def is_admin(request):
    return 'is_staff' in request.session.keys() and request.session['is_staff'] is True

def checkbox_value(request, key):
    return True if key in request.POST.keys() and request.POST[key] == 'on' else False
#### Views
def index(request):
    if is_logged_in(request):
        return dashboard(request)
    return render(request, PAGES.LOGIN_PAGE)


### Login
def login(request):
    if is_logged_in(request):
        return dashboard(request)

    context = {}
    username = request.POST['username']
    password = request.POST['password']

    user = authenticate(username=username, password=password)

    if user is not None:
        user.last_login = timezone.now()
        user.save()

        request.session['user_id'] = user.id
        request.session['username'] = user.username
        request.session['is_staff'] = user.is_staff
        request.session['fullname'] = ('{} {}').format(user.first_name, user.last_name)
        request.session['batch'] = user.batch

        return dashboard(request)

    else:
        context['message'] = PAGES.MESSAGE_LOGIN_FAILED
        context['message_type'] = PAGES.MESSAGE_TYPE_ALERT
        return render(request, PAGES.LOGIN_PAGE, context)


## Logout
def logout(request):
    request.session.flush()
    return render(request, PAGES.LOGIN_PAGE)


## Dashboard: shows progress for each batch
def dashboard(request):
    if not is_logged_in(request) :
        return index(request)

    context = set_context(request)

    if is_admin(request):
        return render(request, PAGES.ADMIN_DASHBOARD_PAGE, context)

    if not is_batch_assigned(request):
        return render(request, PAGES.ERROR_PAGE)


    # get a list of all the posts and if they are annotated or not and view with post_id
    posts, total, annotated = controller.get_annotations_info(request.session['user_id'], request.session['batch'])
    context['posts'] = posts.sample(frac=1)
    context['total'] = total
    context['annotated'] = annotated
    return render(request, PAGES.DASHBOARD_PAGE, context)



def annotate(request, post_id):

    if not is_logged_in(request):
        return index(request)

    if not is_batch_assigned(request):
        return render(request, PAGES.ERROR_PAGE)
    post = controller.get_post_info(post_id)
    context = set_context(request)
    context['post'] = post
    context['annotated'] = False
    context['post_id'] = post_id


    # check if post exists
    annotation = models.Annotation.getAnnotation(request.session['user_id'], post_id)
    if annotation is not None:
        context['comments'] = annotation.comments
        context['post_text'] = annotation.post_text
        context['result'] = annotation.result
        context['issue'] = annotation.issue

    context['total'] = len(models.StudyPost().getBatchPosts(context['batch']))
    context['annotated'] = len(models.Annotation.objects.filter(user=request.session['user_id']))


    return render(request, PAGES.ANNOTATION_PAGE, context)

def save_annotation(request):
    if  not is_logged_in(request):
        return index(request)

    if not is_batch_assigned(request):
        return render(request, PAGES.ERROR_PAGE)    
    user_id = request.session['user_id']
    batch = request.session['batch']
    post_id = request.POST['post_id']
    post_text = request.POST['post_text']
    comments = request.POST['comments']

    result = {}

    issue = request.POST['post_issue']

    for k, v in request.POST.items():
        if k != 'csrfmiddlewaretoken':
            result[k] = v
    print(result)
    

    u = models.User.objects.get(pk=user_id)
    annotation = models.Annotation.getAnnotation(user_id, post_id)
    if annotation == None: #create
        u.annotation_set.create(post_id=post_id, post_text=post_text, result=result,
                                comments=comments, issue= issue, annotation_date=datetime.now())
    else: # update
        annotation.post_text = post_text
        annotation.result = result
        annotation.issue = issue
        annotation.comments = comments
        annotation.save()

    next_pair= controller.get_next_unannotated_pair(user_id, batch)
    if next_pair == None:
        return dashboard(request)

    return annotate(request,  next_pair)

### Admin pages: create user, upload posts  and download annotations
def create_show_users(request):
    if not is_logged_in(request) or not is_admin(request):
        return index(request)

    context = set_context(request)

    if request.method != 'GET':

        user, created = models.User.objects.update_or_create(
            first_name = request.POST['firstname'],
            last_name = request.POST['lastname'],
            username=request.POST['username'],
            email = request.POST['email'],
            batch = request.POST['batch'] if not checkbox_value(request, 'staff') else None,
            is_staff =  checkbox_value(request, 'staff'),
            is_active = checkbox_value(request, 'active'),

        )
        user.set_password(request.POST['password'])
        user.save()

    context['users_data'] = models.User.objects.all()

    return render(request, PAGES.USERS_PAGES, context)

def upload_posts(request):
    if not is_logged_in(request) or not is_admin(request):
        return index(request)
    context = set_context(request)

    if request.method != 'GET':
        csv_file = request.FILES['file']
        if not csv_file.name.endswith('.csv'):
            messages.error(request, "Please upload a .csv file with columns : 'id', 'source', 'issue', 'batch' ")
        else:
            data_set = csv_file.read().decode('UTF-8')
            io_string = io.StringIO(data_set)
            next(io_string)  # to skip header
            for column in csv.reader(io_string, delimiter=','):
                post, created = models.StudyPost.objects.update_or_create(
                    id=column[0],
                    defaults={
                        'source' : column[1],
                        'issue': column[2],
                        'batch': column[3]
                        
                    }
                    
                )
                post.save()
    context['post_data'] = models.StudyPost.objects.all()
    context['count'] = len(models.StudyPost.objects.all())
    print(context['post_data'])
    return render(request, PAGES.UPLOAD_POSTS, context)

def download_annotations(request):
    if  not is_logged_in(request) or not is_admin(request):
        return index(request)
    return controller.export_to_csv(models.Annotation)

def download_users(request):
    if  not is_logged_in(request) or not is_admin(request):
        return index(request)
    return controller.export_to_csv(models.User)

def download_posts(request):
    if not is_logged_in(request) or not is_admin(request):
        return index(request)
    return controller.export_to_csv(models.StudyPost)

def view_annotations(request, batch_num):

    if  not is_logged_in(request) or not is_admin(request):
        return index(request)

    context = set_context(request)
    context['annotations_data'], context['annotated_count'], context['total_count'] = controller.get_all_annotations(batch_num)
    context['batch_num'] = batch_num
    return render(request, PAGES.VIEW_ANNOTATIONS_PAGE, context)

def deactivate_user(request, user_id):
    if  not is_logged_in(request) or not is_admin(request):
        return index(request)

    controller.deactivate_user(user_id)
    return create_show_users(request)

def activate_user(request, user_id):
    if  not is_logged_in(request) or not is_admin(request):
        return index(request)

    controller.activate_user(user_id)
    return create_show_users(request)

def download_annotations_view(request, batch_num):
    if  not is_logged_in(request) or not is_admin(request):
        return index(request)
    ann_df, _, _ =  controller.get_all_annotations(batch_num)
    return controller.export_to_csv(df=ann_df, name='batch{}_annotations'.format(batch_num))
from django.db import models

from django.contrib.auth.models import AbstractUser
from datetime import datetime
from django.db.models import Count
from django.core.exceptions import ObjectDoesNotExist
from django.contrib.postgres.fields import JSONField, ArrayField

class User(AbstractUser):
    batch =  models.IntegerField(null=True)
    def getUser(email):
        return User.objects.filter(username=email)
    def re_deactivate(self, user_id, activate = True):
        u = User.objects.get(id=user_id)
        u.is_active= activate
        u.save()

class StudyPost(models.Model):
    id = models.TextField(primary_key=True)
    #title = models.CharField(max_length=500, null=False, default="Title")
    #selftext = models.CharField(max_length=2000000, null=False, default="Self text")
    source = models.TextField(max_length=10000, null=False, default="Source")
    issue = models.TextField(max_length=500, null=False, default="Issue")
    batch = models.IntegerField(null=True)
    
    def getPost(self, post_id):
        return StudyPost.objects.get(pk=post_id)
    
    def getAllPosts(self):
        return StudyPost.objects.all()
    
    def getBatchPosts(self, batch):
        return StudyPost.objects.filter(batch=batch)
    
class Annotation(models.Model):
    a_id = models.AutoField(primary_key=True)
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    post = models.ForeignKey(StudyPost, on_delete=models.CASCADE, null=True)
    post_text = models.TextField(max_length=10000, null=False, default="Post Text")
    annotation_date = models.DateTimeField(default=datetime.now)
    result = JSONField(default=dict)
    issue = models.CharField(max_length=500, null=False, default="Post Issue")
    comments = models.CharField(max_length=1000, blank=True) 

    def getUserAnnotations(user_id):
        return Annotation.objects.filter(user=user_id)

    def get_num_of_annotations_per_user(user_id):
        result = Annotation.objects.values('post').filter(user=user_id).annotate(
            count=Count('post'))
        return result['post']

    def getAnnotation(user_id, post_id):
        try:
            return Annotation.objects.get(user_id=user_id, post_id=post_id)
        except ObjectDoesNotExist as e:
            return None

    def get_batch_annotations(self, batch):
        try:
            return Annotation.objects.filter(post__batch=batch, user__is_active=True)
        except ObjectDoesNotExist as e:
            return None


from django.contrib import admin

# Register your models here.
from .models import *

admin.site.register(User)
admin.site.register(StudyPost)
admin.site.register(Annotation)

# Generated by Django 3.0.3 on 2021-01-18 08:26

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('study', '0008_auto_20210117_1035'),
    ]

    operations = [
        migrations.AddField(
            model_name='annotation',
            name='post_text',
            field=models.TextField(default='Post Text', max_length=10000),
        ),
    ]

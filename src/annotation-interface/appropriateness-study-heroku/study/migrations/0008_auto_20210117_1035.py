# Generated by Django 3.0.3 on 2021-01-17 10:35

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('study', '0007_auto_20210117_1030'),
    ]

    operations = [
        migrations.RenameField(
            model_name='annotation',
            old_name='results',
            new_name='result',
        ),
    ]

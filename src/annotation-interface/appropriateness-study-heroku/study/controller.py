import pandas as pd
import csv
import ast
import random

from django.http import HttpResponse
from .models import StudyPost, Annotation, User


################# Helpers #################

def qs_to_df(qs):
    if qs is not None:
        return pd.DataFrame(list(qs.values()))
    return None

def get_post_info(post_id):
    #path = "data/study-annotations.csv"
    path = "data/annotation_dataset.csv"
    data = {}
    df = pd.read_csv(path)
    print(post_id)
    print(df.head())
    df = df[df['id'] == post_id]
    print(df.head())
    data['id'] = df['id'].values[0]
    data['source'] = df['source'].values[0]
    data['issue'] = df['issue'].values[0]
    data['batch'] = df['batch'].values[0]
    return data


################# Controller methods #################

def get_annotations_info(user_id, batch):
    print('batch: ', batch)
    post_df = qs_to_df(StudyPost().getBatchPosts(batch))
    post_df.sort_values(['id'], inplace=True)
    post_df.set_index('id', inplace=True)

    annotation_df = qs_to_df(Annotation.getUserAnnotations(user_id)) # annotation_date,synergy, explanation
    if annotation_df is not None and len(annotation_df) > 0:
        annotation_df.set_index('post_id', inplace=True)
        post_df = post_df.join(annotation_df, how='left', rsuffix='_annotator' )
        post_df['result'].fillna('', inplace=True)
        post_df['comments'].fillna('', inplace=True)
        #post_df['issue'].fillna('', inplace=True)
    else:
        post_df['result'] = ''
        post_df['comments'] = ''
        #post_df['issue'] = ''
        post_df['annotation_date'] = None

    post_df.reset_index(inplace=True)

    total = len(post_df)
    annotated = len(annotation_df) if annotation_df is not None else 0

    return post_df, total, annotated

def _add_annotations_info(row, annotations):
    post_id = row['id']
    post_annotations = annotations[annotations['post_id'] == post_id].copy() if (annotations is not None and len(annotations) >0) else None

    if (post_annotations) is not None :
        row['annotations_num'] = len(post_annotations)
        row['result'] = list(post_annotations['result'].values)
        row['users'] = ', '.join([str(x) for x in list(post_annotations['user_id'].values)])
    return row

def get_all_annotations(batch):
    post_df = qs_to_df(StudyPost().getBatchPosts(batch))
    annotations_df = qs_to_df(Annotation().get_batch_annotations(batch))
    post_df['annotations_num'] = 0
    post_df['result'] = ''
    post_df['users'] = ''

    post_df = post_df.apply(_add_annotations_info, axis=1, args=(annotations_df,))
    annotated_count = len(post_df[post_df['annotations_num'] >= 3])
    total = len(annotations_df)
    return post_df, annotated_count, total

def get_unannotated(user_id, batch):
    all_annotations, total, annotated = get_annotations_info(user_id, batch)
    unannotated_df = (all_annotations[all_annotations['result'] == '']).copy()
    return unannotated_df

def get_next_unannotated_pair(user_id, batch):
    unannotated_df = get_unannotated(user_id, batch)
    return unannotated_df.id.values[0] if (len(unannotated_df) > 0) else None

#### Export
def export_to_csv(model_class= None, df=None, name=''):

    data = qs_to_df(model_class.objects.all()) if (model_class is not None) else df.copy()
    response = None

    if data is not None:
        meta = model_class._meta if (model_class is not None) else name
        field_names = list(data.columns.values)

        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename={}.csv'.format(meta)
        writer = csv.writer(response)

        writer.writerow(field_names)
        for _, row in data.iterrows():
            row = writer.writerow([row[field] for field in field_names])

    return response

def deactivate_user(user_id):
    User().re_deactivate(user_id, activate=False)

def activate_user(user_id):
    User().re_deactivate(user_id)
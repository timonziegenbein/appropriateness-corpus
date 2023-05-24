**Step 1. Create database**

Assuming PostgreSQL is installed, the first step is to create a database.

`psql postgres`

`CREATE DATABASE study;`

**Step 2. Update the database connection in settins.py**

Inside `study/settings.py ` update the `DATABASES` entry with the database connection from Step 1. For example:

```
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql_psycopg2',
        'NAME': 'study',
        'USER': 'postgres',
        'PASSWORD': 'postgres',
        'HOST': 'localhost',
        'PORT':''
        
    }
}
```


**Step 2. Make migrations**
Inside the django app folder, run
`python manage.py makemigrations`

**Step 3. Apply migrations to the database**
`python manage.py migrate`

**Step 4. Start the server**
`python manage.py runserver`

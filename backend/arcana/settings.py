"""Django settings for arcana project."""
import os
import tempfile
from pathlib import Path

import dj_database_url
from dotenv import load_dotenv

# WAL mode for concurrent SQLite access (gunicorn + celery)
def _enable_wal(sender, connection, **kwargs):
    if connection.vendor == 'sqlite':
        cursor = connection.cursor()
        cursor.execute('PRAGMA journal_mode=WAL;')
        cursor.execute('PRAGMA busy_timeout=5000;')

from django.db.backends.signals import connection_created
connection_created.connect(_enable_wal)

# Load .env from project root
BASE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BASE_DIR.parent
load_dotenv(PROJECT_ROOT / '.env')

SECRET_KEY = os.getenv('DJANGO_SECRET_KEY', 'dev-insecure-key-change-in-production')

# DEBUG=False is critical — DEBUG=True buffers SSE streams via Werkzeug/Django middleware
DEBUG = False

ALLOWED_HOSTS = ['*']

INSTALLED_APPS = [
    'corsheaders',
    'api',
]

MIDDLEWARE = [
    'corsheaders.middleware.CorsMiddleware',
    'api.middleware.APIKeyMiddleware',
    'whitenoise.middleware.WhiteNoiseMiddleware',
    'django.middleware.common.CommonMiddleware',
    # No CsrfViewMiddleware — API-only backend, no HTML forms
]

CORS_ALLOW_ALL_ORIGINS = True
CORS_ALLOW_HEADERS = [
    'accept', 'authorization', 'content-type', 'user-agent',
    'x-csrftoken', 'x-requested-with', 'x-api-key',
]

ROOT_URLCONF = 'arcana.urls'

# No templates needed — API-only
TEMPLATES = []

# Database — PostgreSQL via DATABASE_URL (Railway), SQLite fallback (local/Docker)
DATABASE_URL = os.getenv('DATABASE_URL')
if DATABASE_URL:
    DATABASES = {
        'default': dj_database_url.parse(DATABASE_URL)
    }
else:
    DB_PATH = os.getenv('DB_PATH', str(BASE_DIR / 'db.sqlite3'))
    DATABASES = {
        'default': {
            'ENGINE': 'django.db.backends.sqlite3',
            'NAME': DB_PATH,
        }
    }

# File upload limit (16MB, matches Flask config)
DATA_UPLOAD_MAX_MEMORY_SIZE = 16 * 1024 * 1024

# Upload folder — configurable for Docker volume mount
UPLOAD_FOLDER = Path(os.getenv('UPLOAD_DIR', str(Path(tempfile.gettempdir()) / 'arcana_uploads')))
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

# Static files — WhiteNoise serves frontend/ at root URL
STATIC_URL = '/static/'
STATIC_ROOT = PROJECT_ROOT / 'staticfiles'
WHITENOISE_ROOT = PROJECT_ROOT / 'frontend'

# Celery / Redis
CELERY_BROKER_URL = os.getenv('CELERY_BROKER_URL', os.getenv('REDIS_URL', 'redis://localhost:6379/0'))
CELERY_RESULT_BACKEND = os.getenv('CELERY_RESULT_BACKEND', os.getenv('REDIS_URL', 'redis://localhost:6379/0'))
CELERY_ACCEPT_CONTENT = ['json']
CELERY_TASK_SERIALIZER = 'json'
CELERY_RESULT_SERIALIZER = 'json'
CELERY_RESULT_EXPIRES = 3600  # 1 hour

# Startup messages
if os.getenv('RUN_MAIN') != 'true':  # Avoid duplicate prints from reloader
    if not os.getenv('OPENAI_API_KEY'):
        print("WARNING: OPENAI_API_KEY not set in environment. LLM calls will fail.")
        print("Set it in .env file or export OPENAI_API_KEY=your-key")
    print(f"Upload folder: {UPLOAD_FOLDER}")

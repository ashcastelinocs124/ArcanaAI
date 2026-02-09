"""Django settings for arcana project."""
import os
import tempfile
from pathlib import Path

from dotenv import load_dotenv

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
    'django.middleware.common.CommonMiddleware',
    # No CsrfViewMiddleware — API-only backend, no HTML forms
]

CORS_ALLOW_ALL_ORIGINS = True

ROOT_URLCONF = 'arcana.urls'

# No templates needed — API-only
TEMPLATES = []

# Database — default SQLite, never used (no models)
DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

# File upload limit (16MB, matches Flask config)
DATA_UPLOAD_MAX_MEMORY_SIZE = 16 * 1024 * 1024

# Upload folder
UPLOAD_FOLDER = Path(tempfile.gettempdir()) / 'arcana_uploads'
UPLOAD_FOLDER.mkdir(exist_ok=True)

# Celery / Redis
CELERY_BROKER_URL = os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/0')
CELERY_RESULT_BACKEND = os.getenv('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0')
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

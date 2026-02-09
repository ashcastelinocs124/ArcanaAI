FROM python:3.12-slim

WORKDIR /app

# Install system deps for gevent + psycopg2
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libc-dev \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create data directory for SQLite + uploads (Docker local)
RUN mkdir -p /data

ENV PYTHONUNBUFFERED=1
ENV PORT=5000

EXPOSE ${PORT}

# Default: migrate, seed trace data, then run gunicorn
CMD cd backend && \
    python manage.py migrate --noinput && \
    python manage.py seed_traces && \
    gunicorn arcana.wsgi:application \
    --bind 0.0.0.0:${PORT} \
    --worker-class gevent \
    --workers 2 \
    --timeout 120 \
    --access-logfile -

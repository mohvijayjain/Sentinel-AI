FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /mlflow

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        && rm -rf /var/lib/apt/lists/*

# Install MLflow and PostgreSQL driver
RUN pip install --no-cache-dir \
    mlflow==3.15.2 \
    SQLAlchemy==2.0.52 \
    alembic==1.19.1 \
    psycopg2-binary==2.9.12

EXPOSE 5000

CMD ["mlflow", "server"]
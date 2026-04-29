FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app


COPY requirements.txt /app/requirements-api.txt
RUN python -m pip install --upgrade pip && \
    pip install -r /app/requirements-api.txt

# Copia o codigo da aplicacao.
COPY . /app

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]

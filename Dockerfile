FROM python:3.12.10-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONWARNINGS=error \
    MPLBACKEND=Agg

WORKDIR /app
COPY requirements-lock.txt /app/requirements-lock.txt
RUN python -m pip install --no-cache-dir --upgrade pip==26.3 \
    && python -m pip install --no-cache-dir -r requirements-lock.txt
COPY . /app
RUN python validate_clean_environment.py
CMD ["python", "validate_clean_environment.py", "--full"]

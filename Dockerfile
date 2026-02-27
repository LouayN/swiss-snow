FROM python:3.10-slim

WORKDIR /app

# Install dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ src/
COPY app.py app.py
COPY pipelines/ pipelines/

# Copy models (3 MB — acceptable to bake into image)
COPY models/ models/

# Copy processed features (small parquets needed for inference)
# Raw data (200+ MB) is excluded via .dockerignore
COPY data/processed/ data/processed/

# Expose both service ports (only one is used per container instance)
EXPOSE 8000 8501

# Default: run FastAPI API server
# Override CMD in docker-compose or Render to run Streamlit instead
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]

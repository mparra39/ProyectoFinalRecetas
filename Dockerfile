FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# libgomp1 : OpenMP threading for FAISS and PyTorch
# gcc / g++ : fallback compilation for llama-cpp-python if no pre-built wheel matches
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps before copying source so this layer is cached on code changes
COPY requirements-inference.txt .
RUN pip install --no-cache-dir -r requirements-inference.txt

# Install the recetas package (path resolution for config.py depends on it)
COPY pyproject.toml .
COPY recetas/ recetas/
RUN pip install --no-cache-dir -e . --no-deps

COPY run_local.py .

EXPOSE 7860

# data/ and models/ must be mounted as volumes — they are not baked into the image.
# Set GRADIO_SERVER_NAME=0.0.0.0 via docker-compose so the app is reachable from the host.
# Remove --no-cnn / --no-llm when the model files are present in the mounted volumes.
CMD ["python", "run_local.py", "--no-cnn", "--no-llm"]

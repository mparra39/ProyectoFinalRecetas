FROM python:3.12-slim

WORKDIR /app

# libgomp1: required by FAISS and PyTorch for OpenMP threading
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies before copying source (cache-friendly layer order)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy package source and install without re-resolving deps
COPY pyproject.toml .
COPY recetas/ recetas/
RUN pip install --no-cache-dir -e . --no-deps

COPY run_local.py .

EXPOSE 7860

# data/ and models/ are expected as volume mounts at runtime.
# Override CMD to enable CNN/LLM when the model files are present.
CMD ["python", "run_local.py", "--no-cnn", "--no-llm"]

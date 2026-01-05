FROM python:3.10-slim

WORKDIR /app

# Install system dependencies (gcc is needed for some python libraries)
RUN apt-get update && apt-get install -y gcc && rm -rf /var/lib/apt/lists/*

# --- CHECKPOINT 1: Install PyTorch separately ---
# This creates a cached layer. If the build fails later, this step is saved.
RUN mkdir -p src/diffusyn && \
    touch src/diffusyn/__init__.py

# --- FIX START: Force Install CPU Torch First ---
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --no-cache-dir

# 1. Copy configuration files first
COPY pyproject.toml README.md ./

# --- CHECKPOINT 2: Install the rest of the dependencies ---
# We use ".[server]" to install dependencies listed in pyproject.toml
RUN pip install --no-cache-dir ".[server]" && \
    rm -rf src

# 3. Copy the actual source code (this changes frequently)
COPY src ./src

# 4. Install the package itself (instantaneous)
RUN pip install --no-cache-dir --no-deps .

# Default command (API)
CMD ["uvicorn", "diffusyn.interface.api:app", "--host", "0.0.0.0", "--port", "8000"]
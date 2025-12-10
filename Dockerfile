# Base Image (Lightweight Python)
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# 1. Install System Dependencies (gcc needed for some python math libs)
RUN apt-get update && apt-get install -y gcc && rm -rf /var/lib/apt/lists/*

# 2. Copy Dependency Definition
COPY pyproject.toml README.md ./

# 3. Install Dependencies (Server Mode)
# This installs all the libraries in pyproject.toml [server]
RUN pip install --no-cache-dir ".[server]"

# 4. Copy Source Code
COPY src ./src

# 5. Install Our Project
# This makes 'import diffusyn' work globally inside the container
RUN pip install --no-cache-dir .

# 6. Default Command (Can be overridden by Docker Compose)
# By default, we launch the API.
CMD ["uvicorn", "diffusyn.interface.api:app", "--host", "0.0.0.0", "--port", "8000"]
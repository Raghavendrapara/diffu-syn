
FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y gcc && rm -rf /var/lib/apt/lists/*

# 1. Copy configuration files
COPY pyproject.toml README.md ./

# 2. Create a dummy source structure to trick pip into installing dependencies
#    This allows us to cache the heavy "pip install" layer.
RUN mkdir -p src/diffusyn && \
    touch src/diffusyn/__init__.py && \
    pip install --no-cache-dir ".[server]" && \
    rm -rf src

# 3. Now copy the actual source code
COPY src ./src

# 4. Install the package itself (lightweight, no dependencies needed as they are already there)
RUN pip install --no-cache-dir --no-deps .

CMD ["uvicorn", "diffusyn.interface.api:app", "--host", "0.0.0.0", "--port", "8000"]
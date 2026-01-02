
FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y gcc && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./


RUN pip install --no-cache-dir ".[server]"

COPY src ./src

RUN pip install --no-cache-dir .

CMD ["uvicorn", "diffusyn.interface.api:app", "--host", "0.0.0.0", "--port", "8000"]
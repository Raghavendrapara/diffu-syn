import os
import uuid

import aiofiles
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from diffusyn.interface.config import settings
from diffusyn.interface.worker import train_model_task, celery_app
from diffusyn.core.pipeline import TabularDiffusion

app = FastAPI(title="Diffu-Syn API (Cloud Native)", version="0.2.0")

# --- Config ---
UPLOAD_DIR = "data/raw"
MODEL_DIR = "data/models"
OUTPUT_DIR = "data/outputs"

for d in [UPLOAD_DIR, MODEL_DIR, OUTPUT_DIR]:
    os.makedirs(d, exist_ok=True)


class GenerateRequest(BaseModel):
    task_id: str
    n_samples: int = 100


@app.post("/train")
async def train(file: UploadFile = File(...), epochs: int = settings.DEFAULT_EPOCHS):
    """
    Async Training Endpoint.
    1. Saves file to disk (or S3 in future).
    2. Pushes job to Redis.
    3. Returns Task ID immediately.
    """
    # 1. Save File
    file_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{file_id}.csv")

    async with aiofiles.open(file_path, 'wb') as out_file:
        while content := await file.read(settings.CHUNK_SIZE_BYTES):
            await out_file.write(content)

    # 2. Define Output Path
    model_path = os.path.join(MODEL_DIR, f"{file_id}.pth")

    # 3. Offload to Worker (Non-blocking)
    task = train_model_task.delay(file_path, model_path, epochs)

    return {
        "message": "Training queued",
        "task_id": task.id,
        "file_id": file_id
    }


@app.get("/status/{task_id}")
def get_status(task_id: str):
    """Check if the worker has finished."""
    result = celery_app.AsyncResult(task_id)
    return {
        "task_id": task_id,
        "status": result.status,
        "result": result.result if result.ready() else None
    }


@app.post("/generate")
def generate(request: GenerateRequest):
    """
    Inference Endpoint.
    """
    # 1. Check if Training is Done
    task_result = celery_app.AsyncResult(request.task_id)
    if not task_result.successful():
        raise HTTPException(status_code=400, detail=f"Training not complete. Status: {task_result.status}")

    model_path = task_result.result.get("model_path")
    if not model_path or not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model file not found.")

    # 2. Load Core Library
    model = TabularDiffusion()
    model.load(model_path)

    # 3. Generate
    df = model.generate(n_samples=request.n_samples)

    # 4. Return CSV
    output_filename = f"syn_{request.task_id}.csv"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    df.write_csv(output_path)

    return FileResponse(output_path, media_type="text/csv", filename=output_filename)
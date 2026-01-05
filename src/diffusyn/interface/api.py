import uuid
import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from diffusyn.interface.config import settings
from diffusyn.interface.worker import train_model_task, generate_data_task, evaluate_data_task, celery_app
from diffusyn.interface.storage import LocalStorage

app = FastAPI(title="Diffu-Syn API", version="0.4.0")

upload_storage = LocalStorage(base_dir=settings.UPLOAD_DIR)
output_storage = LocalStorage(base_dir=settings.OUTPUT_DIR)

class GenerateRequest(BaseModel):
    task_id: str
    n_samples: int = 100

class EvaluateRequest(BaseModel):
    file_id: str
    synthetic_filename: str


@app.post("/train")
async def train(
        file: UploadFile = File(...),
        epochs: int = settings.DEFAULT_EPOCHS
):
    # 1. Generate ID
    file_id = str(uuid.uuid4())
    filename = f"{file_id}.csv"

    # 2. Save using Service
    try:
        saved_path = upload_storage.save_upload(file, filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

    # 3. Define Model Path
    model_filename = f"{file_id}.pth"
    model_path = os.path.join(settings.MODEL_DIR, model_filename)

    # 4. Offload to Worker
    task = train_model_task.delay(filename, model_path, epochs)

    return {
        "message": "Training queued",
        "task_id": task.id,
        "file_id": file_id
    }


@app.get("/status/{task_id}")
def get_status(task_id: str):
    """Check if ANY worker task (Train or Generate) has finished."""
    result = celery_app.AsyncResult(task_id)
    return {
        "task_id": task_id,
        "status": result.status,
        "result": result.result if result.ready() else None
    }


@app.post("/generate")
async def trigger_generation(request: GenerateRequest):
    """
    Async Inference Endpoint.
    """
    # 1. Lookup the Training Task to find the Model
    training_result = celery_app.AsyncResult(request.task_id)

    if not training_result.successful():
        raise HTTPException(status_code=400,
                            detail=f"Training not complete or failed. Status: {training_result.status}")

    full_model_path = training_result.result.get("model_path")
    if not full_model_path:
        raise HTTPException(status_code=404, detail="Model info not found in training result")

    model_filename = os.path.basename(full_model_path)

    # 2. Prepare Output
    job_id = str(uuid.uuid4())
    output_filename = f"syn_{job_id}.csv"

    # 3. Offload to Worker
    task = generate_data_task.delay(
        model_filename=model_filename,
        n_samples=request.n_samples,
        output_filename=output_filename
    )

    return {
        "message": "Generation queued",
        "generation_task_id": task.id,
        "download_url": f"/download/{output_filename}"
    }


@app.post("/evaluate")
async def trigger_evaluation(request: EvaluateRequest):
    """
    Quality Evaluation Endpoint.
    """
    original_filename = f"{request.file_id}.csv"
    
    task = evaluate_data_task.delay(
        original_filename=original_filename,
        synthetic_filename=request.synthetic_filename
    )

    return {
        "message": "Evaluation queued",
        "evaluation_task_id": task.id
    }


@app.get("/download/{filename}")
async def download_file(filename: str):
    try:
        file_path = output_storage.get_local_path(filename)
        return FileResponse(file_path, media_type='text/csv', filename=filename)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not ready or not found.")

from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import shutil
import os
import uuid

# Import our Engine
from diffusyn.engine.trainer import train_model
from diffusyn.engine.inference import generate_synthetic_data

app = FastAPI(title="Diffu-Syn API", version="0.1.0")

# --- Configuration ---
UPLOAD_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
OUTPUT_DIR = "data/outputs"  # New folder for results
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# --- Pydantic Models ---
class TrainRequest(BaseModel):
    file_id: str
    epochs: int = 5
    batch_size: int = 32


class GenerateRequest(BaseModel):
    model_id: str  # The ID returned by the Train step
    n_samples: int = 100


# --- Background Tasks ---
def run_training_task(file_path: str, model_path: str, epochs: int):
    print(f"⚙️ Training started on {file_path}...")
    try:
        # Hardcoded 5 dims for the demo, in prod this would be dynamic
        train_model(data_path=file_path, output_path=model_path, epochs=epochs, input_dim=5)
        print(f"✅ Training Complete. Model saved to {model_path}")
    except Exception as e:
        print(f"❌ Training Failed: {e}")


# --- Endpoints ---

@app.get("/")
def health_check():
    return {"status": "online", "engine": "Diffusion-v1"}


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_id = f"{uuid.uuid4()}.csv"
    file_path = os.path.join(UPLOAD_DIR, file_id)

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"message": "Upload successful", "file_id": file_id}


@app.post("/train")
async def trigger_training(request: TrainRequest, background_tasks: BackgroundTasks):
    file_path = os.path.join(UPLOAD_DIR, request.file_id)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File ID not found")

    # The model ID is just the file ID but with .pth extension
    model_id = request.file_id.replace(".csv", ".pth")
    model_path = os.path.join(PROCESSED_DIR, model_id)

    background_tasks.add_task(run_training_task, file_path, model_path, request.epochs)

    return {"status": "training_started", "model_id": model_id}


@app.post("/generate")
async def generate_data(request: GenerateRequest):
    """
    Generates data and returns the CSV file directly.
    """
    model_path = os.path.join(PROCESSED_DIR, request.model_id)
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model not ready (Is training finished?)")

    output_filename = f"synthetic_{request.model_id.replace('.pth', '.csv')}"
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    try:
        # Generate logic
        generate_synthetic_data(
            model_path=model_path,
            output_csv=output_path,
            n_samples=request.n_samples,
            input_dim=5  # Hardcoded for demo
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

    # Return the file to the browser
    return FileResponse(path=output_path, filename=output_filename, media_type='text/csv')
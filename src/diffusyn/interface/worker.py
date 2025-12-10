import os
from celery import Celery
from diffusyn.core.pipeline import TabularDiffusion

# 1. Setup Celery
celery_app = Celery(
    "diffusyn_worker",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/0"
)


# 2. Define the Heavy Task
@celery_app.task(bind=True)
def train_model_task(self, file_path: str, model_save_path: str, epochs: int = 5):
    """
    This runs on the Worker Node (GPU), not the API Node.
    """
    try:
        print(f"👷 Worker received job: {file_path}")

        # Initialize the Core Logic
        model = TabularDiffusion()

        # Train (The library handles loading the CSV)
        model.fit(file_path, epochs=epochs)

        # Save Artifact
        model.save(model_save_path)

        return {"status": "success", "model_path": model_save_path}

    except Exception as e:
        print(f"❌ Training failed: {e}")
        self.retry(exc=e, countdown=60, max_retries=3)
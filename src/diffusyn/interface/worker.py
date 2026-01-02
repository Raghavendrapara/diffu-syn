import os
from celery import Celery
from diffusyn.core.pipeline import TabularDiffusion
from diffusyn.interface.config import settings
from diffusyn.interface.storage import LocalStorage

celery_app = Celery(
    "diffusyn_worker",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL
)
upload_storage = LocalStorage(base_dir=settings.UPLOAD_DIR)
model_storage  = LocalStorage(base_dir=settings.MODEL_DIR)
output_storage = LocalStorage(base_dir=settings.OUTPUT_DIR)


@celery_app.task(bind=True)
def train_model_task(self, filename: str, model_save_path: str, epochs: int = 5):
    try:
        print(f" Worker received job for file: {filename}")

        file_path = upload_storage.get_local_path(filename)

        # 2. Train
        model = TabularDiffusion()
        model.fit(file_path, epochs=epochs)

        model.save(model_save_path)
        return {"status": "success", "model_path": model_save_path}

    except Exception as e:
        print(f"Training failed: {e}")
        self.retry(exc=e, countdown=60, max_retries=3)


@celery_app.task(bind=True)
def generate_data_task(self, model_filename: str, n_samples: int, output_filename: str):
    try:
        print(f"🔮 Worker generating {n_samples} samples...")

        model_path = model_storage.get_local_path(model_filename, check_exists=True)

        output_path = output_storage.get_local_path(output_filename, check_exists=False)

        # 3. Generate & Save
        model = TabularDiffusion()
        model.load(model_path)
        df = model.generate(n_samples=n_samples)
        df.write_csv(output_path)

        return {"status": "success", "file_id": output_filename}

    except Exception as e:
        print(f" Generation failed: {e}")
        self.retry(exc=e, countdown=60, max_retries=3)
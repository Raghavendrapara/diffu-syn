import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # --- Infrastructure ---
    # 1MB = 1024 * 1024. Configurable via env var: CHUNK_SIZE_BYTES
    CHUNK_SIZE_BYTES: int = 1_048_576

    BASE_DATA_DIR: str = "data"
    UPLOAD_DIR: str = os.path.join(BASE_DATA_DIR, "raw")
    MODEL_DIR: str = os.path.join(BASE_DATA_DIR, "models")
    OUTPUT_DIR: str = os.path.join(BASE_DATA_DIR, "outputs")

    REDIS_URL: str = "redis://localhost:6379/0"

    DEFAULT_EPOCHS: int = 5
    DEFAULT_BATCH_SIZE: int = 32

    class Config:
        env_prefix = "DIFFUSYN_"


# Singleton instance
settings = Settings()

# Ensure directories exist at startup
for directory in [settings.UPLOAD_DIR, settings.MODEL_DIR, settings.OUTPUT_DIR]:
    os.makedirs(directory, exist_ok=True)
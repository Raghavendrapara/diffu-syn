import os
import aiofiles
from typing import Protocol, BinaryIO


class StorageService(Protocol):
    """
    The Interface (Contract).
    Any future storage backend (S3, GCS, Azure) MUST implement these methods.
    """

    async def save_upload(self, file_stream: BinaryIO, filename: str) -> str:
        """Saves an incoming stream and returns a unique identifier/path."""
        ...

    def get_local_path(self, file_id: str) -> str:
        """
        Returns a valid local filesystem path for the Worker to read.
        (If using S3, this method would download the file to a temp folder first).
        """
        ...


class LocalStorage:
    """
    The Implementation for Disk/Volume Storage.
    """

    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)

    async def save_upload(self, upload_file, filename: str) -> str:
        # 1. Define location
        file_path = os.path.join(self.base_dir, filename)

        # 2. Write in chunks (Non-blocking)
        # We assume 'upload_file' behaves like a FastAPI UploadFile (has .read)
        async with aiofiles.open(file_path, 'wb') as out_file:
            while content := await upload_file.read(1024 * 1024):  # 1MB chunks
                await out_file.write(content)

        # 3. Return the absolute path or ID
        return file_path

    def get_local_path(self, filename: str, check_exists: bool = True) -> str:
        path = os.path.join(self.base_dir, filename)

        if check_exists and not os.path.exists(path):
            raise FileNotFoundError(f"File {filename} not found in {self.base_dir}")

        return path
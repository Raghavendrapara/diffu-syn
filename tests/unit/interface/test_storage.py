import pytest
import os
from io import BytesIO
from fastapi import UploadFile
from diffusyn.interface.storage import LocalStorage

@pytest.mark.anyio
async def test_local_storage_save(tmp_path):
    storage = LocalStorage(base_dir=str(tmp_path))
    
    content = b"test data"
    # Mocking UploadFile with an async read
    class MockUploadFile:
        def __init__(self, content):
            self.content = content
            self.read_called = False
        async def read(self, size):
            if self.read_called:
                return b""
            self.read_called = True
            return self.content

    upload_file = MockUploadFile(content)
    
    saved_path = await storage.save_upload(upload_file, "saved.txt")
    
    assert os.path.exists(saved_path)
    with open(saved_path, "rb") as f:
        assert f.read() == content

def test_local_storage_get_path(tmp_path):
    storage = LocalStorage(base_dir=str(tmp_path))
    file_path = tmp_path / "exists.txt"
    file_path.write_text("hello")
    
    assert storage.get_local_path("exists.txt", check_exists=True) == str(file_path)
    
    with pytest.raises(FileNotFoundError):
        storage.get_local_path("missing.txt", check_exists=True)
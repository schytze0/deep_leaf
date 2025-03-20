import os
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

@pytest.fixture
def fake_image_jpg(tmp_path):
    """Create a small dummy .jpg file for valid testing."""
    file_path = tmp_path / "test.jpg"
    with open(file_path, "wb") as f:
        f.write(b"\xFF\xD8\xFF")  # minimal JPEG header bytes
    return file_path

@pytest.fixture
def fake_image_png(tmp_path):
    """Create a small dummy .png file for valid testing."""
    file_path = tmp_path / "test.png"
    with open(file_path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")  # minimal PNG header bytes
    return file_path

def test_invalid_extension(tmp_path):
    # Create a dummy .txt file
    file_path = tmp_path / "test.txt"
    with open(file_path, "w") as f:
        f.write("Hello, world!")

    with open(file_path, "rb") as file:
        response = client.post("/predict/", files={"file": ("test.txt", file, "text/plain")})
    assert response.status_code == 400
    assert "Invalid file type" in response.text

def test_file_too_large(tmp_path):
    # Create a dummy large file
    file_path = tmp_path / "test.jpg"
    with open(file_path, "wb") as f:
        f.write(b"\xFF\xD8\xFF" + b"a" * (6 * 1024 * 1024))  # 6MB 

    with open(file_path, "rb") as file:
        response = client.post("/predict/", files={"file": ("test.jpg", file, "image/jpeg")})
    assert response.status_code == 413
    assert "exceeds" in response.text

def test_valid_jpg(fake_image_jpg):
    with open(fake_image_jpg, "rb") as file:
        response = client.post("/predict/", files={"file": ("test.jpg", file, "image/jpeg")})
    assert response.status_code == 200
    assert "prediction" in response.json()

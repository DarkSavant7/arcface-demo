import insightface
import cv2
import numpy as np
from PIL import Image
import base64
import io
from typing import List, Tuple, Optional


class FaceRecognizer:
  def __init__(self):
    # Load model ArcFace
    self.model = insightface.app.FaceAnalysis(name="buffalo_l")
    self.model.prepare(ctx_id=0)  # Using CPU (ctx_id=0)

  def decode_image(self, image_data: str) -> np.ndarray:
    """Decodes base64 image into numpy array"""
    image_bytes = base64.b64decode(image_data)
    image = Image.open(io.BytesIO(image_bytes))
    return np.array(image)

  def extract_embeddings(self, image: np.ndarray) -> List[np.ndarray]:
    """Extracts embeddings from the image"""
    # Convert RGB to BGR (OpenCV format)
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Face detection and embedding extraction
    faces = self.model.get(image_bgr)
    return [face.embedding for face in faces]

  def get_face_embedding_base64(self, image_data: str) -> Tuple[
    Optional[List[float]], int]:
    """Extracts embeddings from the image and returns the number of faces found."""
    image = self.decode_image(image_data)
    embeddings = self.extract_embeddings(image)

    if len(embeddings) == 0:
      return None, 0
    elif len(embeddings) > 1:
      return None, len(embeddings)

    return embeddings[0].tolist(), 1

  def get_face_embedding_binary(self, image_bytes: bytes) -> Tuple[Optional[List[float]], int]:
    """Extracts embeddings from a binary image (e.g., multipart file content)."""
    image = Image.open(io.BytesIO(image_bytes))
    image = image.convert("RGB")
    image_np = np.array(image)
    embeddings = self.extract_embeddings(image_np)

    if len(embeddings) == 0:
      return None, 0
    elif len(embeddings) > 1:
      return None, len(embeddings)

    return embeddings[0].tolist(), 1


# Global instance
face_recognizer = FaceRecognizer()
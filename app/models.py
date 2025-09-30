from pydantic import BaseModel
from typing import List, Optional
from enum import Enum

class RecognitionStatus(str, Enum):
    SUCCESS = "success"
    MULTIPLE_FACES = "multiple_faces"
    NO_FACES = "no_faces"
    NOT_REGISTERED = "not_registered"

class RegisterRequest(BaseModel):
    name: str
    images: List[str]  # base64 encoded images

class RecognitionResponse(BaseModel):
    status: RecognitionStatus
    name: Optional[str] = None
    similarity: Optional[float] = None
    error: Optional[str] = None

class RecognitionRequest(BaseModel):
    image: str  # base64 encoded image
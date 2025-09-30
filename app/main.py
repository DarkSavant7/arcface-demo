import logging

from fastapi import FastAPI, HTTPException
from app.models import RegisterRequest, RecognitionRequest, RecognitionResponse, \
  RecognitionStatus
from app.face_utils import face_recognizer
from app.database import vector_db
from app.config import SIMILARITY_THRESHOLD

app = FastAPI(title="Face Recognition API")
logger = logging.getLogger(__name__)


@app.get("/")
async def root():
  return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
  return {"message": f"Hello {name}"}


@app.post("/register", response_model=dict)
async def register_person(request: RegisterRequest):
  """
  Registers a new person in the system,
  принимает список изображений в формате base64 и имя человека
  """

  if vector_db.exists_by_name(request.name):
    raise HTTPException(status_code=409,
                        detail=f"Person with name '{request.name}' has already been registered")
  all_embeddings = []

  for image_data in request.images:
    embedding, faces_count = face_recognizer.get_face_embedding(image_data)

    if faces_count == 0:
      logger.warning(f"On the image for the {request.name} found no faces")
      continue
    elif faces_count > 1:
      logger.warning(f"On the image for the {request.name} found more than one face")
      continue

    all_embeddings.append(embedding)

  if not all_embeddings:
    raise HTTPException(status_code=400,
                        detail="Couldn't extract embeddings from the images")

  # Усредняем эмбеддинги для более устойчивого представления
  avg_embedding = [sum(values) / len(values) for values in zip(*all_embeddings)]

  # Сохраняем в базу данных
  vector_db.add_embedding(avg_embedding, request.name)

  return {"status": "success",
          "message": f"Person {request.name} registered successfully",}


@app.post("/recognize", response_model=RecognitionResponse)
async def recognize_person(request: RecognitionRequest):
  """
  Recognizes a person in the system,
  accepts a base64 image and returns the name of the person.
  """
  embedding, faces_count = face_recognizer.get_face_embedding(request.image)

  if faces_count == 0:
    return RecognitionResponse(status=RecognitionStatus.NO_FACES,
                               error="Faces weren't recognized on the image")

  if faces_count > 1:
    return RecognitionResponse(status=RecognitionStatus.MULTIPLE_FACES,
                               error="Found more than one face on the image")

  # Ищем похожие лица в базе данных
  search_results = vector_db.search_similar(embedding)

  if not search_results or search_results[0].score < SIMILARITY_THRESHOLD:
    return RecognitionResponse(status=RecognitionStatus.NOT_REGISTERED,
                               error="Person wasn't found in the database")

  # Возвращаем результат с наибольшей схожестью
  best_match = search_results[0]
  return RecognitionResponse(
      status=RecognitionStatus.SUCCESS,
      name=best_match.payload["name"],
      similarity=best_match.score
  )

@app.delete("/persons/{name}", response_model=dict)
async def delete_person(name: str):
  """
  Delete a person from the system by name
  """
  # Проверим наличие
  if not vector_db.exists_by_name(name):
    raise HTTPException(status_code=404, detail=f"Person '{name}' not found")

  deleted = vector_db.delete_by_name(name)
  return {"status": "success", "message": f"Person '{name}' deleted", "deleted": deleted}


@app.get("/health")
async def health_check():
  """Healthcheck endpoint"""
  return {"status": "healthy"}

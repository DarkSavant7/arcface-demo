from qdrant_client import QdrantClient
from qdrant_client.http import models
from app.config import QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME, EMBEDDING_SIZE
import uuid


class VectorDB:
  def __init__(self):
    self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    self.collection_name = COLLECTION_NAME
    self._create_collection_if_not_exists()

  def _create_collection_if_not_exists(self):
    """Create a collection if it doesn't exist."""
    collections = self.client.get_collections().collections
    collection_names = [collection.name for collection in collections]

    if self.collection_name not in collection_names:
      self.client.create_collection(
          collection_name=self.collection_name,
          vectors_config=models.VectorParams(
              size=EMBEDDING_SIZE,
              distance=models.Distance.COSINE
          )
      )

  def add_embedding(self, embedding: list, name: str):
    """Add embedding to the database."""
    point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, name))
    self.client.upsert(
        collection_name=self.collection_name,
        points=[
          models.PointStruct(
              id=point_id,
              vector=embedding,
              payload={"name": name}
          )
        ]
    )

  def search_similar(self, embedding: list, limit: int = 1):
    """Look for similar embeddings in the database."""
    return self.client.search(
        collection_name=self.collection_name,
        query_vector=embedding,
        limit=limit
    )

  def exists_by_name(self, name: str) -> bool:
    """Checks if a record with the given name exists in the database."""
    result = self.client.scroll(
        collection_name=self.collection_name,
        scroll_filter=models.Filter(
            must=[models.FieldCondition(key="name",
                                        match=models.MatchValue(value=name))]
        ),
        limit=1
    )
    points, _ = result
    return len(points) > 0

  def delete_by_name(self, name: str) -> int:
    """Delete a record with the given name from the database."""
    # Use a filter by payload 'name'
    res = self.client.delete(
        collection_name=self.collection_name,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[models.FieldCondition(key="name", match=models.MatchValue(
                  value=name))]
            )
        )
    )
    # Qdrant delete is asynchronous; could be checked by scrolling
    points, _ = self.client.scroll(
        collection_name=self.collection_name,
        scroll_filter=models.Filter(
            must=[models.FieldCondition(key="name",
                                        match=models.MatchValue(value=name))]
        ),
        limit=1
    )
    return 0 if points else 1


# Global instance
vector_db = VectorDB()
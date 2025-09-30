import os

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "face_embeddings")
EMBEDDING_SIZE = 512  # Embedding size ArcFace
SIMILARITY_THRESHOLD = 0.5  # Similarity threshold for face recognition
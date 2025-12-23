import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import (
    PointStruct,
    ScoredPoint,
    VectorParams,
    Distance,
    Filter,
    FieldCondition,
    MatchAny,
    MatchValue,
)

import logging

logger = logging.getLogger(__name__)

class QdrantVectorDB:
    def __init__(self, collection_name: str, embedding_dim: int, host: str, port: int):
        self.collection_name = collection_name
        self.embedding_dim = embedding_dim
        self.client = QdrantClient(host=host, port=port)

    def create_collection(self, recreate: bool = False):
        if self.client.collection_exists(self.collection_name):
            if recreate:
                self.client.delete_collection(self.collection_name)
            else:
                logger.warning(f"Collection {self.collection_name} already exists, skipping creation")
                return
            
        logger.info(f"Creating collection {self.collection_name}")
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.embedding_dim,
                distance=Distance.COSINE
            )
        )

    def insert(self, embedding: list[float], payload: dict):
        point = PointStruct(id=str(uuid.uuid4()), vector=embedding, payload=payload)
        self.client.upsert(collection_name=self.collection_name, points=[point])


    def search(self, embedding: list[float], top_k: int) -> list[ScoredPoint]:

        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=embedding,
            limit=top_k,
        )
        return results


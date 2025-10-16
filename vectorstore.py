# vectorstore.py
import os
from dotenv import load_dotenv
from typing import List, Dict, Any

load_dotenv()

import pinecone

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENVIRONMENT")  # sometimes called PINECONE_ENV or PINECONE_ENVIRONMENT
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

if not PINECONE_API_KEY or not PINECONE_INDEX_NAME:
    raise EnvironmentError("PINECONE_API_KEY or PINECONE_INDEX_NAME missing in .env")

# initialize pinecone
if PINECONE_ENV:
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
else:
    pinecone.init(api_key=PINECONE_API_KEY)

_index = pinecone.Index(PINECONE_INDEX_NAME)

def store_in_pinecone(chunks: List[str], embeddings: List[List[float]], namespace: str = "") -> None:
    """Upserts chunks+embeddings to Pinecone index in batches."""
    if len(chunks) != len(embeddings):
        raise ValueError("chunks and embeddings must be same length.")

    vectors_to_upsert = []
    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        metadata = {"text": chunk, "chunk_index": i}
        vectors_to_upsert.append((f"chunk-{i}", emb, metadata))

    batch_size = 100
    for i in range(0, len(vectors_to_upsert), batch_size):
        batch = vectors_to_upsert[i:i + batch_size]
        # pinecone upsert expects list of tuples (id, vector, metadata)
        _index.upsert(vectors=batch, namespace=namespace)

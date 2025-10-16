# embedder.py
import os
from dotenv import load_dotenv
from typing import List

load_dotenv()

# import the LangChain Google GenAI chat wrapper
from langchain_google_genai import ChatGoogleGenerativeAI

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

if not GEMINI_API_KEY:
    raise EnvironmentError("GEMINI_API_KEY is not set in environment. Please set it in .env")

# Initialize client - pass model explicitly to satisfy pydantic validation
client = ChatGoogleGenerativeAI(api_key=GEMINI_API_KEY, model=GEMINI_MODEL)

EMBEDDING_MODEL = "textembedding-gecko-001"

def embed_chunks(chunks: List[str]) -> List[List[float]]:
    """Generates embeddings for a list of text chunks using Google GenAI via LangChain wrapper."""
    embeddings = []
    for chunk in chunks:
        # The API shape may vary by langchain-google-genai version; the code below follows your original pattern.
        response = client.embeddings.create(
            input=chunk,
            model=EMBEDDING_MODEL
        )
        # Defensive checks:
        if not hasattr(response, "data") or not response.data:
            raise RuntimeError("Embedding response missing `data` or embeddings.")
        embeddings.append(response.data[0].embedding)
    # quick sanity print
    if embeddings:
        print("Sample embedding length:", len(embeddings[0]))
    return embeddings

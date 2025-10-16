# dataprocessor.py
from pdfreader import read_pdf
from chunker import chunk_pages
from embedder import embed_chunks
from vectorstore import store_in_pinecone
from pinecone import Pinecone
from typing import List

pc = Pinecone(api_key="********-****-****-****-************")
index = pc.Index("llm-retrieval-augmented-generation")
pdf_path = "./resources/HRPolicy.pdf"

def run():
    pages = read_pdf(pdf_path)  # Read HR Policies PDF and extract text
    print(f"Read {len(pages)} pages from the PDF.")
    chunks = chunk_pages(pages)  # Chunk the extracted text into smaller pieces
    print(f"Produced {len(chunks)} chunks.")
    embeddings = embed_chunks(chunks)  # Generate embeddings for each chunk
    print(f"Generated {len(embeddings)} embeddings.")
    store_in_pinecone(chunks, embeddings, namespace="hrpolicy")
    print("Upsert to Pinecone completed.")

if __name__ == "__main__":
    run()

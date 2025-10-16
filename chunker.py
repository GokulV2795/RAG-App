# chunker.py
from typing import List

def chunk_pages(pages: List[str], chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Take PDF pages and break into text chunks with simple overlap."""
    chunks: List[str] = []
    for text in pages:
        if not text:
            continue
        start = 0
        n = len(text)
        while start < n:
            end = min(start + chunk_size, n)
            chunk = text[start:end]

            # Try to break at sentence end if possible (heuristic)
            last_period = chunk.rfind('.')
            if last_period != -1 and end < n and (last_period > int(chunk_size * 0.5)):
                end = start + last_period + 1
                chunk = text[start:end]

            chunks.append(chunk.strip())
            # move start forward; keep overlap
            start = max(end - overlap, end)
    return chunks

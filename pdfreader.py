# pdfreader.py
import os
from PyPDF2 import PdfReader
from typing import List

def read_pdf(pdf_path: str) -> List[str]:
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"The file {pdf_path} does not exist.")
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages.append(text)
    return pages

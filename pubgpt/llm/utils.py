from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List


def read_document(pdf_path: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """
    Read and split into chunks a PDF content.

    Args:
        pdf_path (str): PDF path
        chunk_size (int): Size of each chunk
        chunk_overlap (int): Overlap size for each chunk

    Returns:
        List[str]: Chunks
    """
    doc_reader = PdfReader(pdf_path)
    raw_text: str = ""
    for index, page in enumerate(doc_reader.pages):
        text = page.extract_text()
        if text:
            raw_text += text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separator="\n",
        length_function=len,
    )
    return text_splitter.split_text(raw_text)

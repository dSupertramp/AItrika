from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from typing import List


def read_document(document: str, chunk_size: int, chunk_overlap: int) -> List:
    """
    Read and split from string.

    Args:
        document (str): Article text
        chunk_size (int): Dimension of each chunk
        chunk_overlap (int): Dimension of the overlap for each chunk

    Returns:
        List[str]: List of chunks (splitted document)
    """
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    docs = text_splitter.split_text(document)
    return docs


def read_pdf(pdf_path: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """
    Read a and split from PDF.

    Args:
        pdf_path (str): PDF path

    Returns:
        List[str]: List of chunks (splitted document)
    """
    doc_reader = PdfReader(pdf_path)
    raw_text: str = ""
    for index, page in enumerate(doc_reader.pages):
        text = page.extract_text()
        if text:
            raw_text += text
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return text_splitter.split_text(raw_text)

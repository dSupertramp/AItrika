from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from typing import List


def read_pdf(pdf_path: str) -> None:
    """
    Read a local pdf.

    Args:
        pdf_path (str): PDF path

    Returns:
        PdfReader: PdfReader
    """
    doc_reader = PdfReader(pdf_path)
    return doc_reader


def extract_pdf_content(pdf: PdfReader) -> str:
    """
    Extract the content of the PDF.

    Args:
        pdf (PdfReader): PDF as PdfReader

    Returns:
        str: Raw content of the PDF
    """
    raw_text: str = ""
    for index, page in enumerate(pdf.pages):
        text = page.extract_text()
        if text:
            raw_text += text
    return raw_text


def split_pdf_content(pdf_content: str, chunk_size: int, chunk_overlap: int) -> List:
    """
    Split the content of the PDF into chunks.

    Args:
        pdf_content (str): Raw content of the PDF
        chunk_size (int): Dimension of each chunk
        chunk_overlap (int): Dimension of the overlap for each chunk

    Returns:
        List: List of chunks
    """
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return text_splitter.split_text(pdf_content)

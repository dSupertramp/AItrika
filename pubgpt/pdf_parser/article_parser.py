from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from typing import List


def read_pdf(pdf_path: str) -> None:
    doc_reader = PdfReader(pdf_path)
    return doc_reader


def extract_pdf_content(pdf: PdfReader) -> str:
    raw_text: str = ""
    for index, page in enumerate(pdf.pages):
        text = page.extract_text()
        if text:
            raw_text += text
    return raw_text


def split_pdf_content(pdf_content: str) -> List:
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
    )
    return text_splitter.split_text(pdf_content)

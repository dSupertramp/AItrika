from llama_index.core import Document
from typing import List


def generate_documents(content: str) -> List:
    documents = [Document(text=content, id="pdf")]
    return documents

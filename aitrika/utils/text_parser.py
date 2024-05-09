from llama_index.core import Document
from typing import List


def generate_documents(content: str) -> List:
    """
    Generate input documents for Llamaindex.

    Args:
        content (str): Text

    Returns:
        List: List of chunks as Document
    """
    documents = [Document(text=content, id="pdf")]
    return documents

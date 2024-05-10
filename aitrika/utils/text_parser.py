from llama_index.core import Document
from llama_index.core.node_parser import SimpleNodeParser
from typing import List


def generate_documents(content: str) -> List:
    """
    Generate input documents for Llamaindex.

    Args:
        content (str): Text

    Returns:
        List: List of chunks as Document
    """
    parser = SimpleNodeParser(chunk_size=1024, chunk_overlap=80)
    doc = Document(text=content, id=content.partition("\n")[0])
    documents = parser.get_nodes_from_documents([doc])
    return documents

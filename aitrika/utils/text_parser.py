from llama_index.core import Document
from llama_index.core.node_parser import SimpleNodeParser
from typing import List
from aitrika.config.config import LLMConfig


config = LLMConfig()


def generate_documents(content: str) -> List:
    """
    Generate input documents for LlamaIndex.

    Args:
        content (str): Text

    Returns:
        List: List of chunks as Document
    """
    parser = SimpleNodeParser(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
    )
    doc = Document(text=content, id=content.partition("\n")[0])
    documents = parser.get_nodes_from_documents([doc])
    return documents

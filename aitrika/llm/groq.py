from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core import (
    VectorStoreIndex,
    Settings,
    StorageContext,
    load_index_from_storage,
    Document,
)
from llama_index.vector_stores.lancedb import LanceDBVectorStore
import os
from aitrika.llm.base_llm import BaseLLM
from aitrika.config import config


class GroqLLM(BaseLLM):
    def __init__(
        self, documents: Document, api_key: str, model_name: str = "llama3-70b-8192"
    ):
        self.documents = documents
        self.model_name = model_name
        if not api_key:
            raise ValueError("API key is required for Groq.")
        self.api_key = api_key

    def _build_index(self):
        llm = Groq(model=self.model_name, api_key=self.api_key)
        embed_model = HuggingFaceEmbedding(
            model_name=config.DEFAULT_EMBEDDINGS,
            cache_folder=f"aitrika/rag/embeddings/{config.DEFAULT_EMBEDDINGS.replace('/','_')}",
        )
        Settings.llm = llm
        Settings.embed_model = embed_model
        Settings.chunk_size = config.CHUNK_SIZE
        Settings.chunk_overlap = config.CHUNK_OVERLAP
        Settings.context_window = config.CONTEXT_WINDOW
        Settings.num_output = config.NUM_OUTPUT

        if os.path.exists("aitrika/rag/vectorstores/groq"):
            vector_store = LanceDBVectorStore(uri="aitrika/rag/vectorstores/groq")
            storage_context = StorageContext.from_defaults(
                vector_store=vector_store, persist_dir="aitrika/rag/vectorstores/groq"
            )
            index = load_index_from_storage(storage_context=storage_context)
            parser = SimpleNodeParser()
            new_nodes = parser.get_nodes_from_documents(self.documents)
            index.insert_nodes(new_nodes)
            index = load_index_from_storage(storage_context=storage_context)
        else:
            vector_store = LanceDBVectorStore(uri="aitrika/rag/vectorstores/groq")
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            index = VectorStoreIndex(
                nodes=self.documents, storage_context=storage_context
            )
            index.storage_context.persist(persist_dir="aitrika/rag/vectorstores/groq")
        self.index = index

    def query(self, query: str):
        self._build_index()
        query_engine = self.index.as_query_engine()
        response = query_engine.query(query)
        return str(response).strip()

from llama_index.llms.ollama import Ollama
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
from aitrika.config.config import LLMConfig


class OllamaLLM(BaseLLM):
    config = LLMConfig()

    def __init__(self, documents: Document):
        self.documents = documents

    def _build_index(self):
        llm = Ollama(model=self.model_name, request_timeout=120.0)
        embed_model = HuggingFaceEmbedding(
            model_name=self.config.DEFAULT_EMBEDDINGS,
            cache_folder=f"aitrika/rag/embeddings/{self.config.DEFAULT_EMBEDDINGS.replace('/','_')}",
        )
        Settings.llm = llm
        Settings.embed_model = embed_model
        Settings.chunk_size = self.config.CHUNK_SIZE
        Settings.chunk_overlap = self.config.CHUNK_OVERLAP
        Settings.context_window = self.config.CONTEXT_WINDOW
        Settings.num_output = self.config.NUM_OUTPUT

        if os.path.exists("aitrika/rag/vectorstores/ollama"):
            vector_store = LanceDBVectorStore(uri="aitrika/rag/vectorstores/ollama")
            storage_context = StorageContext.from_defaults(
                vector_store=vector_store, persist_dir="aitrika/rag/vectorstores/ollama"
            )
            index = load_index_from_storage(storage_context=storage_context)
            parser = SimpleNodeParser()
            new_nodes = parser.get_nodes_from_documents(self.documents)
            index.insert_nodes(new_nodes)
            index = load_index_from_storage(storage_context=storage_context)
        else:
            vector_store = LanceDBVectorStore(uri="aitrika/rag/vectorstores/ollama")
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            index = VectorStoreIndex(
                nodes=self.documents, storage_context=storage_context
            )
            index.storage_context.persist(persist_dir="aitrika/rag/vectorstores/ollama")
        self.index = index

    def query(self, query: str):
        self._build_index()
        query_engine = self.index.as_query_engine()
        response = query_engine.query(query)
        return str(response).strip()

from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import (
    VectorStoreIndex,
    Settings,
    StorageContext,
    load_index_from_storage,
    Document,
)
import os
from llm.base_llm import BaseLLM
from yaspin import yaspin


class GroqLLM(BaseLLM):
    model_name: str = "gemma-7b-it"
    embeddings: str = "BAAI/bge-small-en-v1.5"
    chunk_size: int = 1024
    chunk_overlap: int = 80

    def __init__(self, documents: Document, api_key: str):
        self.documents = documents
        self.api_key = api_key

    @yaspin()
    def _build_index(self):
        llm = Groq(model=self.model_name, api_key=self.api_key)
        embed_model = HuggingFaceEmbedding(
            model_name=self.embeddings,
            cache_folder="embeddings/huggingface",
        )
        Settings.llm = llm
        Settings.embed_model = embed_model
        Settings.chunk_size = self.chunk_size
        Settings.chunk_overlap = self.chunk_overlap

        if os.path.exists("vectorstores/groq"):
            storage_context = StorageContext.from_defaults(
                persist_dir="vectorstores/groq"
            )
            index = load_index_from_storage(storage_context=storage_context)
        else:
            storage_context = StorageContext.from_defaults()
            index = VectorStoreIndex.from_documents(
                documents=self.documents,
                storage_context=storage_context,
                show_progress=False,
            )
            index.storage_context.persist(persist_dir="vectorstores/groq")
        self.index = index

    @yaspin()
    def query_model(self, query: str):
        self._build_index()
        query_engine = self.index.as_query_engine()
        response = query_engine.query(query)
        return str(response).strip()

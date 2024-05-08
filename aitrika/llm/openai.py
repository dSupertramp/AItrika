from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbeddings
from llama_index.core import (
    VectorStoreIndex,
    Settings,
    StorageContext,
    load_index_from_storage,
    Document,
)
from dotenv import load_dotenv
import os
from llm.base_llm import BaseLLM
from yaspin import yaspin


class OpenAILLM(BaseLLM):
    load_dotenv()
    model_name = "gpt-3.5-turbo"
    emdedding = "text-embedding-3-small"
    chunk_size: int = 1024
    chunk_overlap: int = 80

    def __init__(self, documents: Document, api_key: str):
        self.documents = documents
        self.api_key = api_key

    @yaspin()
    def _build_index(self):
        llm = OpenAI(model=self.model_name, token=self.api_key)
        embed_model = OpenAIEmbeddings(
            model_name=self.embeddings,
            cache_folder="embeddings/openai",
        )
        Settings.llm = llm
        Settings.embed_model = embed_model
        Settings.chunk_size = self.chunk_size
        Settings.chunk_overlap = self.chunk_overlap

        if os.path.exists("openai"):
            storage_context = StorageContext.from_defaults(persist_dir="openai")
            index = load_index_from_storage(storage_context=storage_context)
        else:
            storage_context = StorageContext.from_defaults()
            index = VectorStoreIndex.from_documents(
                documents=self.documents,
                storage_context=storage_context,
                show_progress=False,
            )
            index.storage_context.persist(persist_dir="openai")
        self.index = index

    @yaspin()
    def query(self, query: str):
        self._build_index()
        query_engine = self.index.as_query_engine()
        response = query_engine.query(query)
        return str(response).strip()

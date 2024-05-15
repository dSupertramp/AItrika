from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbeddings
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core import (
    VectorStoreIndex,
    Settings,
    StorageContext,
    load_index_from_storage,
    Document,
)
import os
from aitrika.llm.base_llm import BaseLLM


class OpenAILLM(BaseLLM):
    model_name = "gpt-3.5-turbo"
    emdedding = "text-embedding-3-small"
    chunk_size: int = 1024
    chunk_overlap: int = 80
    context_window: int = 2048
    num_output: int = 256

    def __init__(self, documents: Document, api_key: str):
        self.documents = documents
        if not api_key:
            raise ValueError("API key is required for OpenAI.")
        self.api_key = api_key

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
        Settings.context_window = self.context_window
        Settings.num_output = self.num_output

        if os.path.exists("vectorstores/openai"):
            storage_context = StorageContext.from_defaults(
                persist_dir="vectorstores/openai"
            )
            index = load_index_from_storage(storage_context=storage_context)
            parser = SimpleNodeParser()
            new_nodes = parser.get_nodes_from_documents(self.documents)
            index.insert_nodes(new_nodes)
            index = load_index_from_storage(storage_context=storage_context)
        else:
            storage_context = StorageContext.from_defaults()
            index = VectorStoreIndex(
                nodes=self.documents, storage_context=storage_context
            )
            index.storage_context.persist(persist_dir="vectorstores/openai")
        self.index = index

    def query(self, query: str):
        self._build_index()
        query_engine = self.index.as_query_engine()
        response = query_engine.query(query)
        return str(response).strip()

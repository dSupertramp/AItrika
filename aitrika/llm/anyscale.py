from llama_index.llms.anyscale import Anyscale
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


class AnyscaleLLM(BaseLLM):
    def __init__(
        self,
        documents: Document,
        api_key: str,
        model_endpoint: str = "mistralai/Mixtral-8x7B-Instruct-v0.1",
    ):
        self.documents = documents
        self.model_endpoint = model_endpoint
        if not api_key:
            raise ValueError("API key is required for AnyScale.")
        self.api_key = api_key

    def _build_index(self):
        llm = Anyscale(model=self.model_endpoint, api_key=self.api_key)
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

        if os.path.exists("aitrika/rag/vectorstores/anyscale"):
            vector_store = LanceDBVectorStore(uri="aitrika/rag/vectorstores/anyscale")
            storage_context = StorageContext.from_defaults(
                vector_store=vector_store,
                persist_dir="aitrika/rag/vectorstores/anyscale",
            )
            index = load_index_from_storage(storage_context=storage_context)
            parser = SimpleNodeParser()
            new_nodes = parser.get_nodes_from_documents(self.documents)
            index.insert_nodes(new_nodes)
            index = load_index_from_storage(storage_context=storage_context)
        else:
            vector_store = LanceDBVectorStore(uri="aitrika/rag/vectorstores/anyscale")
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            index = VectorStoreIndex(
                nodes=self.documents, storage_context=storage_context
            )
            index.storage_context.persist(
                persist_dir="aitrika/rag/vectorstores/anyscale"
            )
        self.index = index

    def query(self, query: str):
        self._build_index()
        query_engine = self.index.as_query_engine()
        response = query_engine.query(query)
        return str(response).strip()

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
import os
from aitrika.llm.base_llm import BaseLLM
from aitrika.utils.loader import loader


class AnyscaleLLM(BaseLLM):
    embeddings: str = "BAAI/bge-small-en-v1.5"
    chunk_size: int = 1024
    chunk_overlap: int = 80

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
            model_name=self.embeddings,
            cache_folder="embeddings/huggingface",
        )

        Settings.llm = llm
        Settings.embed_model = embed_model
        Settings.chunk_size = self.chunk_size
        Settings.chunk_overlap = self.chunk_overlap

        if os.path.exists("vectorstores/anyscale"):
            storage_context = StorageContext.from_defaults(
                persist_dir="vectorstores/anyscale"
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
            index.storage_context.persist(persist_dir="vectorstores/anyscale")
        self.index = index

    @loader(text="Querying")
    def query(self, query: str):
        self._build_index()
        query_engine = self.index.as_query_engine()
        response = query_engine.query(query)
        return str(response).strip()

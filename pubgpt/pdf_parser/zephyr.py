from langchain.embeddings import HuggingFaceHubEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import HuggingFaceHub
from typing import List, Any
from dotenv import load_dotenv
import os


load_dotenv()


def create_embeddings(splitted_text_from_pdf: List) -> Any:
    """
    Create embeddings from chunks for Falcon.

    Args:
        splitted_text_from_pdf (List): List of chunks

    Returns:
        Any: Embeddings
    """
    embeddings = HuggingFaceHubEmbeddings(
        repo_id="sentence-transformers/all-mpnet-base-v2",
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    )
    vectorstore = FAISS.from_texts(texts=splitted_text_from_pdf, embedding=embeddings)
    vectorstore.save_local("vector_db")
    persisted_vectorstore = FAISS.load_local("vector_db", embeddings)
    return persisted_vectorstore


def create_chain(query: str, embeddings: Any) -> None:
    """
    Create chain for Falcon.

    Args:
        query (str): Query
        embeddings (Any): Embeddings
    """
    chain = load_qa_chain(
        llm=HuggingFaceHub(
            repo_id="tiiuae/falcon-7b-instruct",
            huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
        ),
        chain_type="stuff",
    )
    docs = embeddings.similarity_search(query)
    chain.run(input_documents=docs, question=query)


def retriever(query: str, embeddings: Any) -> str:
    """
    Create retriever for Falcon.

    Args:
        query (str): Query
        embeddings (Any): Embeddings

    Returns:
        str: Result of retriever
    """
    retriever = embeddings.as_retriever(search_type="similarity")
    result = RetrievalQA.from_chain_type(
        llm=HuggingFaceHub(
            repo_id="HuggingFaceH4/zephyr-7b-alpha",
            huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
        ),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )
    return result(query)["result"]

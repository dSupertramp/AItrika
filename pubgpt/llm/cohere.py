from langchain.embeddings import CohereEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import Cohere
from typing import List, Any
from dotenv import load_dotenv

load_dotenv()


def create_embeddings(splitted_text: List) -> Any:
    """
    Create embeddings from chunks for Cohere.

    Args:
        splitted_text (List): List of chunks

    Returns:
        Any: Embeddings
    """
    embeddings = CohereEmbeddings()
    vectorstore = FAISS.from_texts(texts=splitted_text, embedding=embeddings)
    vectorstore.save_local("vector_db")
    persisted_vectorstore = FAISS.load_local("vector_db", embeddings)
    return persisted_vectorstore


def retriver(query: str, embeddings: Any) -> str:
    """
    Create retriever for Cohere.

    Args:
        query (str): Query
        embeddings (Any): Embeddings

    Returns:
        str: Result of retriever
    """
    retriever = embeddings.as_retriever(search_type="similarity")
    result = RetrievalQA.from_chain_type(
        llm=Cohere(),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )
    return result(query)["result"]

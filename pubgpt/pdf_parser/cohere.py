from langchain.embeddings import CohereEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import Cohere
from typing import List, Any
from dotenv import load_dotenv

load_dotenv()


def create_embeddings_cohere(splitted_text_from_pdf: List) -> Any:
    """
    Create embeddings from chunks for Cohere.

    Args:
        splitted_text_from_pdf (List): List of chunks

    Returns:
        Any: Embeddings
    """
    embeddings = CohereEmbeddings()
    documents = FAISS.from_texts(texts=splitted_text_from_pdf, embedding=embeddings)
    return documents


def create_cohere_chain(query: str, embeddings: Any) -> None:
    """
    Create chain for Cohere.

    Args:
        query (str): Query
        embeddings (Any): Embeddings
    """
    chain = load_qa_chain(llm=Cohere(), chain_type="stuff")
    docs = embeddings.similarity_search(query)
    chain.run(input_documents=docs, question=query)


def retriever_cohere(query: str, embeddings: Any) -> str:
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

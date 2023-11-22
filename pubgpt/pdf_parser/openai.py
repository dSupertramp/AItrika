from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from typing import List, Any
from dotenv import load_dotenv

load_dotenv()


def create_embeddings_openai(splitted_text_from_pdf: List) -> Any:
    """
    Create embeddings from chunks for OpenAI.

    Args:
        splitted_text_from_pdf (List): List of chunks

    Returns:
        Any: Embeddings
    """
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=splitted_text_from_pdf, embedding=embeddings)
    vectorstore.save_local("vector_db")
    persisted_vectorstore = FAISS.load_local("vector_db", embeddings)
    return persisted_vectorstore


def create_opeanai_chain(query: str, embeddings: Any) -> None:
    """
    Create chain for OpenAI.

    Args:
        query (str): Query
        embeddings (Any): Embeddings
    """
    chain = load_qa_chain(llm=OpenAI(), chain_type="stuff")
    docs = embeddings.similarity_search(query)
    chain.run(input_documents=docs, question=query)


def retriever_openai(query: str, embeddings: Any) -> str:
    """
    Create retriever for OpenAI.

    Args:
        query (str): Query
        embeddings (Any): Embeddings

    Returns:
        str: Result of retriever
    """
    retriever = embeddings.as_retriever(search_type="similarity")
    result = RetrievalQA.from_chain_type(
        llm=OpenAI,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )
    return result(query)["result"]

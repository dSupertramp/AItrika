from langchain.embeddings import CohereEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import Cohere
from typing import List, Any, Tuple
from dotenv import load_dotenv
from .prompts import pre_prompt, associations_prompt
import os

load_dotenv()


def create_embeddings(splitted_text: List) -> CohereEmbeddings:
    """
    Create embeddings from chunks for Cohere.

    Args:
        splitted_text (List): List of chunks

    Returns:
        Any: Embeddings
    """
    embeddings = CohereEmbeddings()
    if os.path.exists("vector_db"):
        vectorstore = FAISS.load_local("vector_db", embeddings)
        return vectorstore
    else:
        vectorstore = FAISS.from_texts(texts=splitted_text, embedding=embeddings)
        vectorstore.save_local("vector_db")
        return vectorstore


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


def get_associations(pairs: List[Tuple[str, str]], embeddings: CohereEmbeddings) -> str:
    """
    Get associations from Cohere.

    Args:
        pairs (List[Tuple[str, str]]): Pairs Gene-Disease
        embeddings (CohereEmbeddings): Embeddings

    Returns:
        str: Response
    """
    pre_prompt_pairs: list = []
    for index, item in enumerate(pairs, 1):
        pre_prompt_pairs.append(
            f"{index}) {item[0][0].strip()} associated with {item[1][0].strip()}?"
        )
    pre_prompt_pairs = "\n".join(pre_prompt_pairs)
    query = associations_prompt.format(pairs=pre_prompt_pairs.strip())
    prompt = pre_prompt.format(query=query)
    retriever = embeddings.as_retriever(search_type="similarity")
    result = RetrievalQA.from_chain_type(
        llm=Cohere(),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )
    return result(prompt)["result"]

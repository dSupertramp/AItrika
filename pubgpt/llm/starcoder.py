from langchain.embeddings import HuggingFaceHubEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
from typing import List, Any, Tuple
from dotenv import load_dotenv
import os
from .prompts import pre_prompt, associations_prompt

load_dotenv()


def create_embeddings(splitted_text: List) -> HuggingFaceHubEmbeddings:
    """
    Create embeddings from chunks for Starcoder.

    Args:
        splitted_text (List): List of chunks

    Returns:
        Any: Embeddings
    """
    embeddings = HuggingFaceHubEmbeddings(
        repo_id="sentence-transformers/all-mpnet-base-v2",
        task="feature-extraction",
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    )
    vectorstore = FAISS.from_texts(texts=splitted_text, embedding=embeddings)
    vectorstore.save_local("vector_db")
    persisted_vectorstore = FAISS.load_local("vector_db", embeddings)
    return persisted_vectorstore


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
            repo_id="bigcode/starcoder",
            huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
        ),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )
    return result(query)["result"]


def get_associations(
    pairs: List[Tuple[str, str]], embeddings: HuggingFaceHubEmbeddings
) -> str:
    """
    Get associations from Starcoder.

    Args:
        pairs (List[Tuple[str, str]]): Pairs Gene-Disease
        embeddings (HuggingFaceHubEmbeddings): Embeddings

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
        llm=HuggingFaceHub(
            repo_id="bigcode/starcoder",
            huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
        ),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )
    return result(prompt)["result"]

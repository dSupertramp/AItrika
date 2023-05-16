from langchain.embeddings import CohereEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import Cohere
from typing import List, Any
from dotenv import load_dotenv

load_dotenv()


def create_embeddings_cohere(splitted_text_from_pdf: List) -> Any:
    embeddings = CohereEmbeddings()
    documents = FAISS.from_texts(texts=splitted_text_from_pdf, embedding=embeddings)
    return documents


def create_cohere_chain(query: str, embeddings: Any) -> None:
    chain = load_qa_chain(llm=Cohere(), chain_type="stuff")
    docs = embeddings.similarity_search(query)
    chain.run(input_documents=docs, question=query)


def retriever_cohere(query: str, embeddings: Any) -> str:
    retriever = embeddings.as_retriever(search_type="similarity")
    result = RetrievalQA.from_chain_type(
        llm=Cohere(),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )
    return result(query)["result"]

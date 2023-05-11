from PyPDF2 import PdfReader
from langchain.embeddings import CohereEmbeddings, OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI, Cohere
from dotenv import load_dotenv
from typing import List, Any


load_dotenv()


def read_pdf(pdf_path: str) -> None:
    doc_reader = PdfReader(pdf_path)
    return doc_reader


def extract_pdf_content(pdf: PdfReader) -> str:
    raw_text: str = ""
    for index, page in enumerate(pdf.pages):
        text = page.extract_text()
        if text:
            raw_text += text
    return raw_text


def split_pdf_content(pdf_content: str) -> List:
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
    )
    return text_splitter.split_text(pdf_content)


def create_embeddings_openai(splitted_text_from_pdf: List) -> Any:
    embeddings = OpenAIEmbeddings()
    documents = FAISS.from_texts(texts=splitted_text_from_pdf, embedding=embeddings)
    return documents


def create_embeddings_cohere(splitted_text_from_pdf: List) -> Any:
    embeddings = CohereEmbeddings()
    documents = FAISS.from_texts(texts=splitted_text_from_pdf, embedding=embeddings)
    return documents


def create_opeanai_chain(query: str, embeddings: Any) -> None:
    chain = load_qa_chain(llm=OpenAI(), chain_type="stuff")
    docs = embeddings.similarity_search(query)
    chain.run(input_documents=docs, question=query)


def create_cohere_chain(query: str, embeddings: Any) -> None:
    chain = load_qa_chain(llm=Cohere(), chain_type="stuff")
    docs = embeddings.similarity_search(query)
    chain.run(input_documents=docs, question=query)


def retriever_openai(query: str, embeddings: Any) -> str:
    retriever = embeddings.as_retriever(search_type="similarity")
    result = RetrievalQA.from_chain_type(
        llm=OpenAI,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )
    return result(query)["result"]


def retriever_cohere(query: str, embeddings: Any) -> str:
    retriever = embeddings.as_retriever(search_type="similarity")
    result = RetrievalQA.from_chain_type(
        llm=Cohere(),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )
    return result(query)["result"]


if __name__ == "__main__":
    pdf_path = "../input/Breast cancer genes: beyond BRCA1 and BRCA2.pdf"
    pdf = read_pdf(pdf_path)
    pdf_content = extract_pdf_content(pdf=pdf)
    splitted_text_from_pdf = split_pdf_content(pdf_content=pdf_content)
    ####
    query = "Is BRCA1 associated with breast cancer?"
    embeddings = create_embeddings_cohere(splitted_text_from_pdf=splitted_text_from_pdf)
    print(retriever_cohere(query=query, embeddings=embeddings))

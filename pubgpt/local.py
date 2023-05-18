from pdf_parser.utils import (
    read_pdf,
    extract_pdf_content,
    split_pdf_content,
)

# from pdf_parser.openai import create_embeddings_openai, retriever_openai
from pdf_parser.cohere import create_embeddings_cohere, retriever_cohere


if __name__ == "__main__":
    pdf_path = "input/Breast cancer genes: beyond BRCA1 and BRCA2.pdf"
    query = "Is BRCA1 associated with breast cancer?"
    pdf = read_pdf(pdf_path)
    pdf_content = extract_pdf_content(pdf=pdf)
    splitted_text_from_pdf = split_pdf_content(
        pdf_content=pdf_content, chunk_size=1000, chunk_overlap=200
    )
    ##########################################################

    # embeddings = create_embeddings_openai(splitted_text_from_pdf=splitted_text_from_pdf)
    # print(retriever_openai(query=query, embeddings=embeddings))
    embeddings = create_embeddings_cohere(splitted_text_from_pdf=splitted_text_from_pdf)
    print(retriever_cohere(query=query, embeddings=embeddings))

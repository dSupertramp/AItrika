from pdf_parser.utils import (
    read_pdf,
    extract_pdf_content,
    split_pdf_content,
)

from pdf_parser.falcon import create_embeddings, retriever


if __name__ == "__main__":
    pdf_path = "input/Breast cancer genes_ beyond BRCA1 and BRCA2.pdf"
    query = "Is BRCA1 associated with breast cancer?"
    pdf = read_pdf(pdf_path)
    pdf_content = extract_pdf_content(pdf=pdf)
    splitted_text_from_pdf = split_pdf_content(
        pdf_content=pdf_content, chunk_size=1000, chunk_overlap=200
    )
    ##########################################################
    embeddings = create_embeddings(splitted_text_from_pdf=splitted_text_from_pdf)
    print(retriever(query=query, embeddings=embeddings))

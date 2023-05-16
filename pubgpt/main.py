from online_parser.article_parser import (
    parse_article,
    extract_genes_and_diseases,
    extract_mesh_terms,
    extract_other_terms,
)


from pdf_parser.article_parser import (
    read_pdf,
    extract_pdf_content,
    split_pdf_content,
)

# from pdf_parser.openai import create_embeddings_openai, retriever_openai
from pdf_parser.cohere import create_embeddings_cohere, retriever_cohere


# from llm.openai import get_associations
from llm.cohere import get_associations


def online_parser(document_id: str):
    paper_id, title, abstract, document = parse_article(document_id=document_id)
    gene_df, disease_df, pairs = extract_genes_and_diseases(document_id=document_id)
    mesh_terms = extract_mesh_terms(document_id=document_id)
    other_terms = extract_other_terms(document_id=document_id)
    ## result_openai = get_associations(document=document, pairs=pairs)
    result_cohere = get_associations(
        document=document, document_id=document_id, pairs=pairs
    )
    print(result_cohere)


def pdf_parser(pdf_path: str, query: str):
    pdf = read_pdf(pdf_path)
    pdf_content = extract_pdf_content(pdf=pdf)
    splitted_text_from_pdf = split_pdf_content(pdf_content=pdf_content)

    # embeddings = create_embeddings_openai(splitted_text_from_pdf=splitted_text_from_pdf)
    # print(retriever_openai(query=query, embeddings=embeddings))
    embeddings = create_embeddings_cohere(splitted_text_from_pdf=splitted_text_from_pdf)
    print(retriever_cohere(query=query, embeddings=embeddings))


if __name__ == "__main__":
    document_id = "32819603"
    pdf_path = "input/Breast cancer genes: beyond BRCA1 and BRCA2.pdf"
    query = "Is BRCA1 associated with breast cancer?"
    # online_parser(document_id=document_id)
    pdf_parser(pdf_path=pdf_path, query=query)

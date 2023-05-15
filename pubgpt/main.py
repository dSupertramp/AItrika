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
    create_embeddings_cohere,
    retriever_cohere,
    create_embeddings_openai,
    retriever_openai,
)


# from llm.openai import get_associations
from llm.cohere import get_associations


def online_parser():
    document_id = "32819603"
    paper_id, title, abstract, document = parse_article(document_id=document_id)
    gene_df, disease_df, pairs = extract_genes_and_diseases(document_id=document_id)
    mesh_terms = extract_mesh_terms(document_id=document_id)
    other_terms = extract_other_terms(document_id=document_id)
    ## result_openai = get_associations(document=document, pairs=pairs)
    result_cohere = get_associations(
        document=document, document_id=document_id, pairs=pairs
    )
    print(result_cohere)


def pdf_parser():
    pdf_path = "input/Breast cancer genes: beyond BRCA1 and BRCA2.pdf"
    pdf = read_pdf(pdf_path)
    pdf_content = extract_pdf_content(pdf=pdf)
    splitted_text_from_pdf = split_pdf_content(pdf_content=pdf_content)
    query = """
    Is BRCA1 associated with breast cancer?
    """
    embeddings = create_embeddings_cohere(splitted_text_from_pdf=splitted_text_from_pdf)
    print(retriever_cohere(query=query, embeddings=embeddings))


if __name__ == "__main__":
    # online_parser()
    pdf_parser()

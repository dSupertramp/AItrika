from online_parser.article_parser import (
    parse_article,
    extract_genes_and_diseases,
    extract_mesh_terms,
    extract_other_terms,
)


from pdf_parser.utils import (
    read_pdf,
    extract_pdf_content,
    split_pdf_content,
)

# from pdf_parser.openai import create_embeddings_openai, retriever_openai
from pdf_parser.cohere import create_embeddings_cohere, retriever_cohere


# from llm.openai import get_associations
from llm_parser.starcoder import get_associations
from llm_parser.cohere import get_associations


def online_parser(pubmed_id: str):
    paper_id, title, abstract, document = parse_article(pubmed_id=pubmed_id)
    gene_df, disease_df, pairs = extract_genes_and_diseases(pubmed_id=pubmed_id)
    mesh_terms = extract_mesh_terms(pubmed_id=pubmed_id)
    other_terms = extract_other_terms(pubmed_id=pubmed_id)
    ######################################################

    # result_openai, result_cohere

    result_starcoder = get_associations(
        document=document, pubmed_id=pubmed_id, pairs=pairs
    )

    print(result_starcoder)


def pdf_parser(pdf_path: str, query: str):
    pdf = read_pdf(pdf_path)
    pdf_content = extract_pdf_content(pdf=pdf)
    splitted_text_from_pdf = split_pdf_content(
        pdf_content=pdf_content, chunk_size=1000, chunk_overlap=200
    )

    # embeddings = create_embeddings_openai(splitted_text_from_pdf=splitted_text_from_pdf)
    # print(retriever_openai(query=query, embeddings=embeddings))
    embeddings = create_embeddings_cohere(splitted_text_from_pdf=splitted_text_from_pdf)
    print(retriever_cohere(query=query, embeddings=embeddings))


if __name__ == "__main__":
    pubmed_id = "32819603"
    pdf_path = "input/Breast cancer genes: beyond BRCA1 and BRCA2.pdf"
    query = "Is BRCA1 associated with breast cancer?"
    online_parser(pubmed_id=pubmed_id)
    # pdf_parser(pdf_path=pdf_path, query=query)

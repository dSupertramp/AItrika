from online_parser.article_parser import (
    parse_article,
    extract_genes_and_diseases,
    extract_mesh_terms,
    extract_other_terms,
)


# from llm.openai import get_associations
from llm.cohere import get_associations


if __name__ == "__main__":
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
